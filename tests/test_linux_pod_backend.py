"""linux-pod worker backend (RunPod training rollouts). Runs only on Linux; on macOS run it in a
Linux container with scripts/check_linux_pod_backend.sh (default caps = RunPod-like: no netns,
so the seccomp socket filter is what blocks the network)."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

from research_env.core import worker
from research_env.core.worker import execute

LINUX = sys.platform.startswith('linux')


@unittest.skipUnless(LINUX, 'linux-pod backend requires Linux')
class LinuxPodBackend(unittest.TestCase):
    def setUp(self):
        worker.linux_pod_capabilities.cache_clear()
        self.caps = worker.linux_pod_capabilities()
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        base.chmod(0o755)
        self.ws = base/'workspace'
        self.ws.mkdir()
        self.private = base/'private'
        self.private.mkdir(mode=0o700)
        self.private.chmod(0o700)
        (self.private/'labels.json').write_text('{"reference_bpm": 72}')

    def tearDown(self):
        self.tmp.cleanup()

    def run_py(self, code, ws=None, **kw):
        ws = ws or self.ws
        (ws/'probe.py').write_text(code)
        return execute(ws, [sys.executable, 'probe.py'], timeout_s=kw.pop('timeout_s', 60),
                       cpus=1, memory_gb=kw.pop('memory_gb', 2), backend='linux-pod', **kw)

    def test_runs_and_reports_capabilities(self):
        r = self.run_py('import os; print(os.getuid()); open("out.txt", "w").write("ok")')
        self.assertTrue(r['execution_valid'], r)
        self.assertIn(r['network_isolation'], ('netns', 'seccomp'))
        self.assertEqual(r['uid_dropped'], self.caps['uid_drop'])
        self.assertFalse(r['scored_resource_isolation'])
        self.assertEqual((self.ws/'out.txt').read_text(), 'ok')     # workspace stays writable
        if self.caps['uid_drop']:
            self.assertEqual(int(r['stdout']), worker.sandbox_ids(self.ws)[0])

    def test_network_blocked(self):
        code = ('import socket\ntry:\n    socket.create_connection(("1.1.1.1", 53), timeout=3)\n'
                'except OSError as e:\n    print("blocked", type(e).__name__)\nelse:\n    print("OPEN")\n')
        r = self.run_py(code)
        self.assertTrue(r['execution_valid'], r)
        self.assertIn('blocked', r['stdout'])
        if self.caps['network'] == 'seccomp':
            self.assertIn('PermissionError', r['stdout'])
        # Local IPC still works (multiprocessing, numerical libraries).
        r = self.run_py('import socket; a, b = socket.socketpair(); a.send(b"x"); print(b.recv(1))')
        self.assertTrue(r['execution_valid'], r)

    def test_private_dir_and_sibling_workspace_unreadable(self):
        if not self.caps['uid_drop']:
            self.skipTest('not running as root; uid drop unavailable')
        r = self.run_py(f'open({str(self.private/"labels.json")!r}).read()')
        self.assertFalse(r['execution_valid'])
        self.assertIn('PermissionError', r['stderr'])
        sibling = Path(self.tmp.name)/'sibling'
        sibling.mkdir()
        self.assertTrue(self.run_py('open("secret.py", "w").write("best artifact")', ws=sibling)['execution_valid'])
        self.assertNotEqual(worker.sandbox_ids(sibling), worker.sandbox_ids(self.ws))
        r = self.run_py(f'import os; print(os.listdir({str(sibling)!r}))')
        self.assertFalse(r['execution_valid'])
        self.assertIn('PermissionError', r['stderr'])

    def test_memory_cap_pod_limits_and_timeout(self):
        r = self.run_py('b = bytearray(3 * 1024**3)', memory_gb=0.5)
        self.assertFalse(r['execution_valid'])
        self.assertIn('MemoryError', r['stderr'])
        os.environ['PHYSIOLOGY_SANDBOX_MAX_MEMORY_GB'] = '0.25'
        try:
            r = self.run_py('print("ok")', memory_gb=4)
            self.assertEqual(r['limits']['memory_gb'], 0.25)
        finally:
            del os.environ['PHYSIOLOGY_SANDBOX_MAX_MEMORY_GB']
        r = self.run_py('import time; time.sleep(30)', timeout_s=1)
        self.assertEqual(r['failure_reason'], 'wall_budget_exhausted')

    def test_streaming_protocol(self):
        code = 'import sys, json\nfor line in sys.stdin:\n    print(json.dumps({"echo": json.loads(line)["x"]}), flush=True)\n'
        r = self.run_py(code, requests=[{'x': 1}, {'x': 2}])
        self.assertTrue(r['execution_valid'], r)
        self.assertEqual(r['responses'], [{'echo': 1}, {'echo': 2}])


class FilterProgram(unittest.TestCase):
    def test_jump_targets_stay_inside_program(self):
        for machine in worker._SECCOMP_ARCH:
            program = worker._socket_filter_program(machine)
            for i, (code, jt, jf, _) in enumerate(program):
                if code in (0x15, 0x35):
                    self.assertLess(i+1+max(jt, jf), len(program))
            self.assertEqual(program[-1], (0x06, 0, 0, 0x80000000))   # last instruction kills

    @staticmethod
    def run_bpf(program, data):
        """Minimal classic-BPF evaluator for the opcodes the filter uses (LD abs, JEQ, JGE, RET)."""
        pc, acc = 0, 0
        while True:
            code, jt, jf, k = program[pc]
            if code == 0x20:
                acc, pc = data[k], pc+1
            elif code in (0x15, 0x35):
                hit = acc == k if code == 0x15 else acc >= k
                pc += 1+(jt if hit else jf)
            elif code == 0x06:
                return k

    def test_filter_semantics_per_arch(self):
        allow, deny, kill = 0x7FFF0000, 0x00050000 | 13, 0x80000000
        for machine, (arch, socket_nr, uring_nr, x32) in worker._SECCOMP_ARCH.items():
            program = worker._socket_filter_program(machine)
            call = lambda nr, a0=0, audit=arch: self.run_bpf(program, {0: nr, 4: audit, 16: a0})
            self.assertEqual(call(0), allow, machine)                       # read/write etc.
            self.assertEqual(call(socket_nr, 1), allow, machine)            # AF_UNIX
            for family in (2, 10, 17):                                      # AF_INET, AF_INET6, AF_PACKET
                self.assertEqual(call(socket_nr, family), deny, machine)
            self.assertEqual(call(uring_nr), deny, machine)                 # io_uring_setup
            self.assertEqual(call(0, audit=0x40000003), kill, machine)      # foreign arch (i386)
            if x32:
                self.assertEqual(call(0x40000000 | socket_nr, 2), kill, machine)


class NonLinux(unittest.TestCase):
    @unittest.skipIf(LINUX, 'only meaningful off Linux')
    def test_refuses_off_linux(self):
        worker.linux_pod_capabilities.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'requires Linux'):
                execute(Path(tmp), ['true'], timeout_s=5, backend='linux-pod')


if __name__ == '__main__':
    unittest.main()
