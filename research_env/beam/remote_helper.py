"""Stdlib-only helper copied into each participant sandbox (never imported there
from the repo). Runs under the sandbox's Python 3.13.

Subcommands:
  apply   TAR DELETIONS_JSON WORKDIR           unpack workspace delta into WORKDIR
  collect MANIFEST_JSON OUT_TAR WORKDIR MAXB   tar files that differ from MANIFEST; print deletions
  run     FSIZE CPU_S -- CMD...                exec CMD with rlimits
  bridge  CHAN FSIZE CPU_S -- CMD...           causal file-channel <-> stdin/stdout relay

Bridge protocol (Beam process stdin is upload-then-EOF, not interactive):
the controller writes CHAN/req-N.json then CHAN/req-N.ready; the bridge writes one
JSON line to the candidate's stdin, reads exactly one line back and atomically
publishes CHAN/resp-N.json. Request N+1 does not exist in the sandbox until the
controller has read response N. CHAN/eof ends the stream; any extra stdout is an
error. Failures are written to CHAN/error and exit code 3.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tarfile
import time

MAX_LINE = 1_000_000


def _limits(fsize, cpu_s):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    return apply


def _safe_rel(name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(x in ('..', '') for x in p.parts):
        raise ValueError(f'unsafe path {name!r}')
    return str(p)


def _hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def apply(tar_path, deletions_path, workdir):
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    for rel in json.loads(Path(deletions_path).read_text()):
        target = root/_safe_rel(rel)
        if target.is_file() and not target.is_symlink():
            target.unlink()
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            _safe_rel(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError('only regular files allowed')
        tar.extractall(root, filter='data')
    os.unlink(tar_path)
    return 0


def collect(manifest_path, out_tar, workdir, max_bytes):
    root = Path(workdir)
    before = json.loads(Path(manifest_path).read_text())
    seen, total = set(), 0
    with tarfile.open(out_tar, 'w:gz') as tar:
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                print(json.dumps({'error': 'symlink_in_workspace'}))
                return 2
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            seen.add(rel)
            if before.get(rel) == _hash(path):
                continue
            total += path.stat().st_size
            if total > max_bytes:
                print(json.dumps({'error': 'workspace_sync_too_large'}))
                return 2
            tar.add(path, arcname=rel, recursive=False)
    print(json.dumps({'deleted': sorted(set(before)-seen), 'bytes': total}))
    return 0


def run(fsize, cpu_s, cmd):
    _limits(fsize, cpu_s)()
    os.execvp(cmd[0], cmd)


def bridge(chan, fsize, cpu_s, cmd):
    chan = Path(chan)
    child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             preexec_fn=_limits(fsize, cpu_s), start_new_session=True)

    def fail(reason):
        (chan/'error.tmp').write_text(reason)
        os.replace(chan/'error.tmp', chan/'error')
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except OSError:  # already gone (ESRCH, or EPERM for a reaped group on macOS)
            pass
        child.wait()
        return 3

    i = 0
    try:
        while True:
            if (chan/f'req-{i}.ready').exists():
                child.stdin.write((chan/f'req-{i}.json').read_bytes())
                child.stdin.flush()
                line = child.stdout.readline(MAX_LINE+1)
                if not line or len(line) > MAX_LINE:
                    return fail('Missing or oversized streaming response')
                (chan/f'resp-{i}.tmp').write_bytes(line)
                os.replace(chan/f'resp-{i}.tmp', chan/f'resp-{i}.json')
                i += 1
            elif (chan/'eof').exists():
                child.stdin.close()
                if child.stdout.read(1):
                    return fail('Extra streaming output')
                return child.wait()
            elif child.poll() is not None:
                return fail('Missing or oversized streaming response')
            else:
                time.sleep(0.01)
    except (BrokenPipeError, OSError) as exc:
        return fail(f'Streaming pipe error: {exc.__class__.__name__}')


def main(argv):
    op = argv[0]
    if op == 'apply':
        return apply(*argv[1:4])
    if op == 'collect':
        return collect(argv[1], argv[2], argv[3], int(argv[4]))
    sep = argv.index('--')
    if op == 'run':
        return run(int(argv[1]), int(argv[2]), argv[sep+1:])
    if op == 'bridge':
        return bridge(argv[1], int(argv[2]), int(argv[3]), argv[sep+1:])
    raise SystemExit(f'unknown op {op}')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
