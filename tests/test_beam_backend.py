"""Beam backend tests against a mocked beam-client SDK (no network, no account).

The fake SDK maps sandbox paths onto a local temp directory and runs processes
locally, so the real in-sandbox helper (sync + causal streaming bridge) executes.
"""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest import mock

from research_env.beam import sandbox as beam_sandbox
from research_env.beam.config import BeamConfig
from research_env.beam.sandbox import execute_beam, _unpack_back
from research_env.beam import scorer_app, upload_data, snapshots
from research_env.core import worker
from research_env.core.common import digest, tree_hashes

ENTRY = Path(beam_sandbox.__file__).parents[1]/'core'/'inference_entry.py'


class FakeStream:
    def __init__(self, path):
        self.path = path

    def read(self):
        return Path(self.path).read_text(errors='replace') if Path(self.path).exists() else ''


class FakeProcess:
    def __init__(self, popen, out, err):
        self.popen, self.stdout, self.stderr = popen, FakeStream(out), FakeStream(err)
        self.killed = False

    def status(self):
        code = self.popen.poll()
        return (-1, 'running') if code is None else (code, 'exited')

    def wait(self, timeout=None):
        return self.popen.wait(timeout=timeout)

    def kill(self):
        self.killed = True
        try:
            os.killpg(self.popen.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.popen.wait()


class FakeFS:
    def __init__(self, inst):
        self.inst = inst

    def create_directory(self, p):
        self.inst.local(p).mkdir(parents=True, exist_ok=True)

    def write_text(self, p, text, encoding='utf-8', mode=0o644):
        self.inst.local(p).parent.mkdir(parents=True, exist_ok=True)
        self.inst.local(p).write_text(text)

    def read_text(self, p, encoding='utf-8'):
        return self.inst.local(p).read_text()

    def upload_file(self, local_path, p):
        if self.inst.fail_upload:
            raise ConnectionError('simulated upload failure')
        self.inst.uploads.append((local_path, p))
        if p.endswith('in.tar.gz'):
            with tarfile.open(local_path) as tar:
                self.inst.uploaded_members.append(sorted(tar.getnames()))
        self.inst.local(p).parent.mkdir(parents=True, exist_ok=True)
        self.inst.local(p).write_bytes(Path(local_path).read_bytes())

    def download_file(self, p, local_path):
        Path(local_path).write_bytes(self.inst.local(p).read_bytes())

    def wait_for(self, p, timeout=60, *, size=None):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.inst.local(p).exists():
                return object()
            time.sleep(0.005)
        raise TimeoutError(p)


class FakeProcessManager:
    def __init__(self, inst):
        self.inst = inst

    def exec(self, *args, cwd='/workspace', env=None, stdin=None):
        args = [sys.executable if a == 'python' else str(self.inst.local(a)) if a.startswith('/') else a
                for a in args]
        local_cwd = self.inst.local(cwd)
        local_cwd.mkdir(parents=True, exist_ok=True)
        n = len(self.inst.processes)
        out, err = self.inst.base/f'.out{n}', self.inst.base/f'.err{n}'
        with open(out, 'wb') as o, open(err, 'wb') as e:
            popen = subprocess.Popen(args, cwd=local_cwd, stdout=o, stderr=e, stdin=subprocess.DEVNULL,
                                     env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, start_new_session=True)
        proc = FakeProcess(popen, out, err)
        self.inst.processes.append((args, proc))
        return proc


class FakeInstance:
    def __init__(self, base):
        self.base = Path(base)
        self.sandbox_id = 'sb-fake'
        self.fs, self.process = FakeFS(self), FakeProcessManager(self)
        self.terminated = 0
        self.processes, self.uploads, self.uploaded_members = [], [], []
        self.fail_upload = False

    def local(self, p):
        return self.base/p.lstrip('/')

    def terminate(self):
        self.terminated += 1
        for _, proc in self.processes:
            if proc.popen.poll() is None:
                proc.kill()
        return True

    def create_image_from_filesystem(self):
        return 'img-reset-1'


class FakeSDK:
    """Stands in for `import beam`."""
    def __init__(self, base):
        self.base, self.created, self.instances = Path(base), [], []
        self.fail_create = False
        sdk = self

        class Sandbox:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                sdk.created.append(kwargs)

            def create(self):
                if sdk.fail_create:
                    raise RuntimeError('simulated create failure')
                inst = FakeInstance(sdk.base/f'sb{len(sdk.instances)}')
                sdk.instances.append(inst)
                return inst

        class Image:
            @classmethod
            def from_id(cls, image_id):
                return ('image', image_id)

        self.Sandbox, self.Image = Sandbox, Image
        self.Volume = lambda name, mount_path: _Rec('Volume', name=name, mount_path=mount_path)
        self.CloudBucketConfig = lambda **kw: _Rec('CloudBucketConfig', **kw)
        self.CloudBucket = lambda name, mount_path, config: _Rec('CloudBucket', name=name, mount_path=mount_path, config=config)
        self.functions = []

        def function(**kwargs):
            def wrap(fn):
                sdk.functions.append(kwargs)
                return _Rec('Function', fn=fn, **kwargs)
            return wrap
        self.function = function


class _Rec:
    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)

    def __repr__(self):
        return f'{self.kind}({self.__dict__})'


class BeamTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.workspace = base/'workspace'
        self.workspace.mkdir()
        (self.workspace/'train.py').write_text('print("hi")\n')
        (self.workspace/'old.txt').write_text('remove me\n')
        self.sdk = FakeSDK(base/'remote')
        self.cfg = BeamConfig(participant_image_id='img-test', snapshot_registry=base/'registry',
                              stream_poll_s=0.02, ttl_margin_s=60)

    def run_beam(self, command, **kw):
        kw.setdefault('timeout_s', 20)
        return execute_beam(self.workspace, command, config=kw.pop('config', self.cfg), sdk=self.sdk, **kw)


class SandboxParameterTests(BeamTestCase):
    def test_network_blocked_limits_capped_no_secrets_public_mount_only(self):
        result = self.run_beam(['python', '-c', 'print(1)'], cpus=16, memory_gb=64, timeout_s=30)
        self.assertTrue(result['execution_valid'], result)
        kw = self.sdk.created[0]
        self.assertIs(kw['block_network'], True)
        self.assertNotIn('allow_list', kw)  # block_network must not be combined with an allow list
        self.assertEqual(kw['secrets'], [])
        self.assertIs(kw['sync_local_dir'], False)
        self.assertIs(kw['docker_enabled'], False)
        self.assertEqual(kw['ports'], [])
        self.assertEqual(kw['cpu'], 8)
        self.assertEqual(kw['memory'], 32768)  # MiB
        self.assertEqual(kw['keep_warm_seconds'], 30 + 60)
        self.assertEqual(kw['image'], ('image', 'img-test'))
        self.assertEqual([(v.kind, v.name, v.mount_path) for v in kw['volumes']],
                         [('Volume', 'physiology-public', '/public')])
        self.assertEqual(kw['env']['HOME'], '/work')
        self.assertEqual(kw['env']['OMP_NUM_THREADS'], '8')
        self.assertEqual(result['limits'], {'cpus': 8, 'memory_gb': 32, 'timeout_s': 30})
        self.assertFalse(result['public_mount_read_only'])  # Volume has no RO flag
        self.assertFalse(result['scored_resource_isolation'])
        self.assertEqual(self.sdk.instances[0].terminated, 1)

    def test_cloud_bucket_mode_is_read_only_and_uses_secret_names(self):
        cfg = BeamConfig(participant_image_id='img-test', public_mount_mode='cloud_bucket',
                         public_bucket='my-public', snapshot_registry=self.cfg.snapshot_registry)
        result = self.run_beam(['python', '-c', 'pass'], config=cfg)
        (mount,) = self.sdk.created[0]['volumes']
        self.assertEqual((mount.kind, mount.name, mount.mount_path), ('CloudBucket', 'my-public', '/public'))
        self.assertIs(mount.config.read_only, True)
        self.assertEqual(mount.config.access_key, 'PHYSIOLOGY_PUBLIC_BUCKET_ACCESS_KEY')
        self.assertTrue(result['public_mount_read_only'])

    def test_private_storage_never_mounted(self):
        self.run_beam(['python', '-c', 'pass'])
        self.assertNotIn('physiology-private', repr(self.sdk.created))
        self.assertNotIn('/private', repr(self.sdk.created))
        with self.assertRaises(ValueError):
            BeamConfig(public_volume='same', private_volume='same').validate()
        with self.assertRaises(ValueError):
            BeamConfig(public_mount_path='/private/x').validate()
        # Defence in depth: even a config that bypassed validate() cannot mount the private volume.
        sneaky = BeamConfig(public_volume='physiology-private', private_volume='other')
        with self.assertRaises(RuntimeError):
            beam_sandbox.public_mounts(_SameName(sneaky), self.sdk)

    def test_missing_image_and_symlinks_rejected_before_any_sandbox(self):
        cfg = BeamConfig(snapshot_registry=self.cfg.snapshot_registry)
        with self.assertRaises(ValueError):
            self.run_beam(['python', '-c', 'pass'], config=cfg)
        (self.workspace/'link').symlink_to('/etc/passwd')
        with self.assertRaises(ValueError):
            self.run_beam(['python', '-c', 'pass'])
        self.assertEqual(self.sdk.created, [])


class _SameName:
    """Config view whose public volume name equals the private one (bypasses validate)."""
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, name):
        if name == 'private_volume':
            return 'physiology-private'
        return getattr(self._cfg, name)


class ExecutionTests(BeamTestCase):
    def test_command_output_and_workspace_sync_both_ways(self):
        code = ("from pathlib import Path; import sys; Path('out').mkdir(); Path('out/model.txt').write_text('w=3');"
                "Path('old.txt').unlink(); print('stdout-line'); print('stderr-line', file=sys.stderr)")
        result = self.run_beam(['python', '-c', code])
        self.assertTrue(result['execution_valid'], result)
        self.assertEqual(result['exit_code'], 0)
        self.assertIn('stdout-line', result['stdout'])
        self.assertIn('stderr-line', result['stderr'])
        self.assertEqual((self.workspace/'out'/'model.txt').read_text(), 'w=3')
        self.assertFalse((self.workspace/'old.txt').exists())
        self.assertTrue((self.workspace/'train.py').exists())
        self.assertEqual(result['backend'], 'beam')

    def test_nonzero_exit_is_invalid(self):
        result = self.run_beam(['python', '-c', 'raise SystemExit(3)'])
        self.assertFalse(result['execution_valid'])
        self.assertEqual(result['exit_code'], 3)
        self.assertEqual(result['failure_reason'], 'worker_failed')
        self.assertEqual(self.sdk.instances[0].terminated, 1)

    def test_timeout_kills_process_and_terminates_sandbox(self):
        start = time.monotonic()
        result = self.run_beam(['python', '-c', 'import time; time.sleep(60)'], timeout_s=1.5)
        self.assertLess(time.monotonic() - start, 15)
        self.assertFalse(result['execution_valid'])
        self.assertEqual(result['failure_reason'], 'wall_budget_exhausted')
        inst = self.sdk.instances[0]
        self.assertTrue(inst.processes[-2][1].killed or inst.processes[-2][1].popen.poll() is not None)
        self.assertEqual(inst.terminated, 1)

    def test_sdk_failure_mid_run_is_invalid_and_cleaned_up(self):
        orig_create = self.sdk.Sandbox.create

        def create(this):
            inst = orig_create(this)
            inst.fail_upload = True
            return inst
        with mock.patch.object(self.sdk.Sandbox, 'create', create):
            result = self.run_beam(['python', '-c', 'pass'])
        self.assertFalse(result['execution_valid'])
        self.assertIn('ConnectionError', result['failure_reason'])
        self.assertEqual(self.sdk.instances[0].terminated, 1)

    def test_sandbox_create_failure_is_invalid(self):
        self.sdk.fail_create = True
        result = self.run_beam(['python', '-c', 'pass'])
        self.assertFalse(result['execution_valid'])
        self.assertIn('simulated create failure', result['failure_reason'])

    def test_terminate_exception_does_not_mask_result(self):
        with mock.patch.object(FakeInstance, 'terminate', side_effect=RuntimeError('gone')):
            result = self.run_beam(['python', '-c', 'pass'])
        self.assertTrue(result['execution_valid'])
        self.assertEqual(result['cleanup_error'], 'terminate_failed: RuntimeError')

    def test_sync_back_rejects_traversal(self):
        bad = Path(self.tmp.name)/'bad.tar.gz'
        with tarfile.open(bad, 'w:gz') as tar:
            info = tarfile.TarInfo('../escape.txt')
            info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))
        with self.assertRaises(ValueError):
            _unpack_back(bad, [], self.workspace, 10**6)
        with self.assertRaises(ValueError):
            _unpack_back(_empty_tar(self.tmp.name), ['../../etc/passwd'], self.workspace, 10**6)
        self.assertFalse((Path(self.tmp.name)/'escape.txt').exists())


def _empty_tar(d):
    p = Path(d)/'empty.tar.gz'
    with tarfile.open(p, 'w:gz'):
        pass
    return p


class StreamingTests(BeamTestCase):
    def stage(self, inference_source):
        root = self.workspace.parent/'replay'
        (root/'artifact').mkdir(parents=True)
        (root/'artifact'/'inference.py').write_text(inference_source)
        (root/'entry.py').write_text(ENTRY.read_text())
        self.workspace = root

    def test_causal_request_response_with_state(self):
        # Hand-checkable: predict returns the cumulative sum of request values and
        # how many request files exist in the sandbox channel when it is called.
        chan = self.sdk.base/'sb0'/self.cfg.control_dir.lstrip('/')/'chan'
        self.stage(f'''
from pathlib import Path
def predict(artifact_dir, request):
    total = (request['state'] or 0) + request['x']
    visible = len(list(Path({str(chan)!r}).glob('req-*.json')))
    return {{'total': total, 'visible': visible, 'state': total}}
''')
        result = self.run_beam(['python', '-I', 'entry.py'], requests=iter([{'x': 1}, {'x': 2}, {'x': 4}]))
        self.assertTrue(result['execution_valid'], result)
        self.assertEqual([r['total'] for r in result['responses']], [1, 3, 7])
        self.assertEqual([r['visible'] for r in result['responses']], [1, 2, 3])  # never sees a future request
        self.assertEqual(self.sdk.instances[0].terminated, 1)

    def test_extra_output_invalidates(self):
        self.stage('import atexit, sys\natexit.register(lambda: print("junk"))\n'
                   'def predict(a, r):\n    return {"state": None}\n')
        result = self.run_beam(['python', '-I', 'entry.py'], requests=iter([{'x': 1}]))
        self.assertFalse(result['execution_valid'])
        self.assertIn('Extra streaming output', result['failure_reason'])

    def test_crash_mid_stream_invalidates(self):
        self.stage('def predict(a, r):\n    raise SystemExit(1)\n')
        result = self.run_beam(['python', '-I', 'entry.py'], requests=iter([{'x': 1}, {'x': 2}]))
        self.assertFalse(result['execution_valid'])
        self.assertIn('Missing or oversized streaming response', result['failure_reason'])
        self.assertEqual(result['responses'], [])

    def test_stream_timeout_terminates(self):
        self.stage('import time\ndef predict(a, r):\n    time.sleep(60)\n')
        result = self.run_beam(['python', '-I', 'entry.py'], requests=iter([{'x': 1}]), timeout_s=1.5)
        self.assertFalse(result['execution_valid'])
        self.assertEqual(result['failure_reason'], 'wall_budget_exhausted')
        self.assertEqual(self.sdk.instances[0].terminated, 1)


class WorkerDispatchTests(unittest.TestCase):
    def test_worker_routes_beam_without_touching_docker(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch('research_env.beam.sandbox.execute_beam', return_value={'backend': 'beam'}) as eb, \
                mock.patch.object(worker, '_docker_allocation') as docker_alloc:
            out = worker.execute(d, ['python', '-c', 'pass'], timeout_s=5, backend='beam', image='img-x')
        self.assertEqual(out, {'backend': 'beam'})
        docker_alloc.assert_not_called()
        self.assertEqual(eb.call_args.kwargs['image'], 'img-x')
        self.assertEqual(eb.call_args.kwargs['cpus'], 8)

    def test_unknown_backend_still_rejected(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            worker.execute(d, ['true'], timeout_s=5, backend='nope')


class SnapshotTests(BeamTestCase):
    def test_reset_image_then_delta_upload(self):
        image_id = snapshots.create_reset_image(self.workspace, 'img-test', config=self.cfg, sdk=self.sdk)
        self.assertEqual(image_id, 'img-reset-1')
        self.assertEqual(self.sdk.instances[0].terminated, 1)
        record = json.loads((self.cfg.snapshot_registry/'img-reset-1.json').read_text())
        self.assertEqual(set(record['workspace_hashes']), {'train.py', 'old.txt'})
        self.assertTrue(self.sdk.created[0]['block_network'])
        # Episode: one file changed, one added. Only those two are uploaded.
        (self.workspace/'train.py').write_text('print("changed")\n')
        (self.workspace/'new.py').write_text('x = 1\n')
        # The fake reset image does not carry files; pre-populate the episode sandbox to mimic it.
        orig_create = self.sdk.Sandbox.create

        def create(this):
            inst = orig_create(this)
            (inst.local('/work')).mkdir(parents=True)
            (inst.local('/work/old.txt')).write_text('remove me\n')
            return inst
        with mock.patch.object(self.sdk.Sandbox, 'create', create):
            result = self.run_beam(['python', '-c', 'print(open("old.txt").read().strip())'], image=image_id)
        self.assertTrue(result['execution_valid'], result)
        self.assertEqual(self.sdk.instances[1].uploaded_members, [['new.py', 'train.py']])
        self.assertIn('remove me', result['stdout'])


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.artifact = base/'artifact'
        self.artifact.mkdir()
        (self.artifact/'inference.py').write_text('def predict(a, r):\n    return {"state": None}\n')
        self.private = base/'private'
        (self.private/'pulse').mkdir(parents=True)
        (self.private/'pulse'/'manifest.json').write_text('{}')

    def test_returns_only_score_dict(self):
        data, dig = scorer_app.pack_artifact(self.artifact)
        seen = {}

        def score_fn(family, artifact_dir, manifest, **kw):
            seen.update(kw, family=family, manifest=manifest, hashes=tree_hashes(artifact_dir))
            return {'execution_valid': True, 'primary_loss': 4.5, 'failure_reason': None, 'metrics': {'mae': 4.5},
                    'resource_use': {}, 'predictions': {'w1': 70.0}}
        out = scorer_app.score_payload(data, dig, 'pulse/manifest.json', family='pulse', profile='fast',
                                       wall_s=60, private_root=self.private, participant_image_id='img-p',
                                       score_fn=score_fn)
        self.assertEqual(out, {'execution_valid': True, 'primary_loss': 4.5, 'failure_reason': None,
                               'metrics': {'mae': 4.5}, 'resource_use': {}})
        self.assertEqual(seen['backend'], 'beam')  # candidate code runs in participant sandboxes
        self.assertEqual(seen['image'], 'img-p')
        self.assertEqual(digest(seen['hashes']), dig)

    def test_tampered_artifact_or_manifest_escape_is_invalid(self):
        data, dig = scorer_app.pack_artifact(self.artifact)
        never = mock.Mock()
        out = scorer_app.score_payload(data, '0'*64, 'pulse/manifest.json', family='pulse', profile='fast',
                                       wall_s=60, private_root=self.private, participant_image_id='i', score_fn=never)
        self.assertFalse(out['execution_valid'])
        self.assertIn('hash mismatch', out['failure_reason'])
        out = scorer_app.score_payload(data, dig, '../artifact/inference.py', family='pulse', profile='fast',
                                       wall_s=60, private_root=self.private, participant_image_id='i', score_fn=never)
        self.assertFalse(out['execution_valid'])
        never.assert_not_called()

    def test_nonfinite_loss_never_valid(self):
        data, dig = scorer_app.pack_artifact(self.artifact)
        out = scorer_app.score_payload(data, dig, 'pulse/manifest.json', family='pulse', profile='fast', wall_s=1,
                                       private_root=self.private, participant_image_id='i',
                                       score_fn=lambda *a, **k: {'execution_valid': True, 'primary_loss': float('nan')})
        self.assertFalse(out['execution_valid'])
        self.assertIsNone(out['primary_loss'])

    def test_scorer_function_has_private_volume_participants_do_not(self):
        sdk = FakeSDK(self.tmp.name)
        cfg = BeamConfig(participant_image_id='img-p')
        with mock.patch('research_env.beam.image.participant_image', return_value='scorer-image'):
            fn = scorer_app.build_scorer(cfg, sdk)
        kw = sdk.functions[0]
        self.assertEqual([(v.name, v.mount_path) for v in kw['volumes']], [('physiology-private', '/private')])
        self.assertEqual(kw['secrets'], ['BEAM_TOKEN'])
        self.assertEqual(kw['retries'], 0)
        self.assertEqual(fn.fn, scorer_app._remote_entry)
        mounts = beam_sandbox.public_mounts(cfg, sdk)
        self.assertNotIn('physiology-private', [m.name for m in mounts])

    def test_submit_rejects_malformed_scorer_reply(self):
        scorer = mock.Mock()
        scorer.remote.return_value = {'predictions': {'w': 1}}
        out = scorer_app.submit_artifact(scorer, self.artifact, 'pulse/manifest.json', family='pulse',
                                         profile='fast', wall_s=1)
        self.assertFalse(out['execution_valid'])
        self.assertNotIn('predictions', out)


class UploadPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'prepared'
        (self.root/'public'/'train').mkdir(parents=True)
        (self.root/'private'/'test').mkdir(parents=True)
        (self.root/'public'/'train'/'manifest.json').write_text(json.dumps({'entries': [{'records': [{'id': 'a'}]}]}))
        (self.root/'private'/'test'/'manifest.json').write_text(json.dumps({'entries': [{'records': [{'reference_bpm': 70}]}]}))

    def test_public_and_private_go_to_separate_volumes(self):
        cmds = upload_data.plan_uploads(self.root, 'pulse', BeamConfig())
        cps = [c for c in cmds if c[1] == 'cp']
        self.assertEqual(cps, [['beam', 'cp', str((self.root/'public').resolve()), 'beam://physiology-public/pulse'],
                               ['beam', 'cp', str((self.root/'private').resolve()), 'beam://physiology-private/pulse']])

    def test_labels_in_public_tree_refused(self):
        (self.root/'public'/'train'/'leak.json').write_text(json.dumps({'records': [{'reference_bpm': 1}]}))
        with self.assertRaises(ValueError):
            upload_data.plan_uploads(self.root, 'pulse', BeamConfig())

    def test_dry_run_executes_nothing(self):
        with mock.patch('subprocess.run') as run, mock.patch('sys.stdout', io.StringIO()), \
                mock.patch('sys.stderr', io.StringIO()):
            self.assertEqual(upload_data.main([str(self.root), '--dataset', 'pulse']), 0)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
