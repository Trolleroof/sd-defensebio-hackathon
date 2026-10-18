"""Trusted, persistent research budget and tamper-evident experiment ledger."""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import time
from research_env.core.common import canonical, digest, tree_hashes
from research_env.core.worker import execute


class Session:
    def __init__(self, root, wall_s=7200, development_calls=30, clock=time.time):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        # One episode's calls are sequential but may arrive on different threads (the GRPO trainer
        # uses asyncio.to_thread); sqlite's own locking still serializes writers across processes.
        self.db = sqlite3.connect(self.root/'controller.sqlite', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
        with self.db:
            for key, value in {'started': clock(), 'wall_s': min(wall_s, 7200),
                               'development_calls': min(development_calls, 30),
                               'used': 0, 'selected': None, 'closed': False, 'checkpoints': []}.items():
                self.db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)', (key, json.dumps(value)))

    def get(self, key):
        return json.loads(self.db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()[0])

    def put(self, key, value):
        self.db.execute('UPDATE settings SET value=? WHERE key=?', (json.dumps(value), key))

    def remaining(self):
        return max(0., self.get('started')+self.get('wall_s')-self.clock())

    def events(self):
        return [json.loads(row[0]) for row in self.db.execute('SELECT payload FROM events ORDER BY seq')]

    def append(self, event):
        events = self.events()
        body = {**event, 'event_id': f'e{len(events):05}', 'timestamp': self.clock(),
                'previous_sha256': events[-1]['sha256'] if events else None}
        body['sha256'] = digest(body)
        self.db.execute('INSERT INTO events(payload) VALUES (?)', (canonical(body).decode(),))
        return body

    def freeze_checkpoints(self):
        elapsed = self.clock()-self.get('started')
        with self.db:
            done = self.get('checkpoints')
            for minute in (15, 30, 60, 120):
                if minute not in done and elapsed >= minute*60:
                    self.append({'kind': 'checkpoint', 'minute': minute, 'candidate_id': self.get('selected'),
                                 'scoring_feedback': 'private_only'})
                    done.append(minute)
            self.put('checkpoints', done)

    def select(self, candidate_id, evidence, reason, final=False):
        self.freeze_checkpoints()
        if self.get('closed') or self.remaining() <= 0:
            raise ValueError('Session closed or wall budget exhausted')
        if not isinstance(evidence, list) or not reason.strip():
            raise ValueError('Selection needs evidence and rationale')
        known = {e.get('candidate_id') for e in self.events() if e['kind'] == 'candidate'}
        if candidate_id not in known:
            raise ValueError('Unknown candidate')
        with self.db:
            self.put('selected', candidate_id)
            self.put('closed', final)
            return self.append({'kind': 'final_selection' if final else 'selection',
                                'candidate_id': candidate_id, 'evidence_ids': evidence, 'reason': reason})

    def run(self, workspace, command, decision, *, seed, config, backend='docker', image=None,
            development=False):
        self.freeze_checkpoints()
        if self.root == Path(workspace).resolve() or self.root.is_relative_to(Path(workspace).resolve()):
            raise ValueError('Private controller must be outside worker workspace')
        required = {'hypothesis', 'expected_result', 'evidence_ids', 'code_diff'}
        if not required <= set(decision) or any(not decision[k] for k in required-{'evidence_ids'}):
            raise ValueError('Substantive decision required')
        known_ids = {e['event_id'] for e in self.events()}
        if not isinstance(decision['evidence_ids'], list) or any(e not in known_ids for e in decision['evidence_ids']):
            raise ValueError('Unknown pre-action evidence ID')
        source = tree_hashes(workspace)
        candidate_id = digest(source)
        with self.db:
            self.db.execute('UPDATE settings SET value=value WHERE key="used"')  # acquire write lock before budget check
            if self.get('closed') or self.remaining() <= 0:
                raise ValueError('Session closed or wall budget exhausted')
            if development and self.get('used') >= self.get('development_calls'):
                raise ValueError('Development budget exhausted')
            if development:
                self.put('used', self.get('used')+1)
            snapshot = self.root/'sources'/candidate_id
            snapshot.mkdir(parents=True, exist_ok=True)
            for name in source:
                if Path(name).suffix in ('.py', '.json', '.txt', '.lock', '.md') and (Path(workspace)/name).stat().st_size <= 1024*1024:
                    target = snapshot/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        target.write_bytes((Path(workspace)/name).read_bytes())
            event = self.append({'kind': 'experiment', 'candidate_id': candidate_id,
                                 'source_hashes': source, 'command': command, 'decision': decision,
                                 'config': config, 'seed': seed, 'development': development})
        try:
            result = execute(workspace, command, timeout_s=self.remaining(), backend=backend,
                             image=image, on_tick=self.freeze_checkpoints)
        except Exception as exc:
            result = {'execution_valid': False, 'failure_reason': f'{type(exc).__name__}: {exc}'}
        self.freeze_checkpoints()
        with self.db:
            self.append({'kind': 'result', 'experiment_id': event['event_id'], 'result': result})
        return event, result

    def export(self, path, outer_split):
        if outer_split not in ('agent_training', 'agent_development', 'agent_final_evaluation'):
            raise ValueError('Declare independent outer task split')
        events = self.events()
        previous = None
        for event in events:
            body = {k: v for k, v in event.items() if k != 'sha256'}
            if event['previous_sha256'] != previous or digest(body) != event['sha256']:
                raise ValueError('Ledger chain corrupted')
            previous = event['sha256']
        Path(path).write_text(''.join(canonical({'outer_split': outer_split, 'event': e}).decode()+'\n' for e in events))
