"""Participant protocol. Provider is caller supplied; no credentials are discovered."""
from pathlib import Path
import shutil
import sys
import time
from research_env.core.common import canonical, digest, invalid_score, read, tree_hashes, write
from research_env.core.session import Session
from research_env.core.evaluation import evaluate
from research_env.tasks.registry import identify


class ResearchEnvironment:
    def __init__(self, controller, workspace, development_manifest, *, backend='docker', image=None):
        self.session = Session(controller)
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.session.root.is_relative_to(self.workspace):
            raise ValueError('Controller inside participant workspace')
        self.development_manifest = Path(development_manifest).resolve()
        if self.development_manifest.is_relative_to(self.workspace):
            raise ValueError('Private development labels inside participant workspace')
        self.backend, self.image = backend, image
        self.task = identify(self.development_manifest)
        identity = {'task': self.task, 'manifest_sha256': digest(read(self.development_manifest)),
                    'backend': backend, 'image': image}
        existing = self.session.db.execute('SELECT value FROM settings WHERE key="task_identity"').fetchone()
        if existing and __import__('json').loads(existing[0]) != identity:
            raise ValueError('Cannot change task, dataset, backend or image while resuming')
        with self.session.db:
            self.session.db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)',
                                    ('task_identity', __import__('json').dumps(identity)))
        if not self.session.events():
            task = self.workspace/f'research_env/tasks/{self.task}/TASK.md'
            with self.session.db:
                self.session.append({'kind': 'initial_evidence', 'content': {
                    'task': task.read_text() if task.exists() else f'{self.task}; CPU only; see published task contract',
                    'interface': 'physiology-research-v1', 'wall_budget_s': self.session.get('wall_s'),
                    'development_calls': self.session.get('development_calls')}})

    def observe(self):
        s = self.session
        return {'interface': 'physiology-research-v1', 'task': self.task, 'remaining_wall_s': s.remaining(),
                'development_calls_remaining': s.get('development_calls')-s.get('used'),
                'selected_candidate_id': s.get('selected'), 'closed': s.get('closed'),
                'history': [{k: v for k, v in e.items() if k not in ('source_hashes', 'hashes')} for e in s.events()],
                'protocol': {'edit': ['path', 'text'], 'command': ['argv', 'decision', 'seed'],
                             'evaluate': ['decision'], 'select': ['candidate_id', 'reason', 'evidence_ids'],
                             'submit': ['candidate_id', 'reason', 'evidence_ids']}}

    def act(self, action, *, model_call_id=None):
        start = time.monotonic()
        with self.session.db:
            action_id = self.session.transcript.append('action_start', {'call_id':model_call_id,'action':action,
                'operation':action.get('operation') if isinstance(action,dict) else None})
        before = len(self.session.events())
        try:
            result = self._act(action)
        except BaseException as exc:
            interrupted = isinstance(exc,(KeyboardInterrupt,SystemExit))
            with self.session.db:
                self.session.transcript.append('action_result', {'action_trace_id':action_id,'call_id':model_call_id,
                    'wall_s':time.monotonic()-start,'status':'cancelled' if interrupted else 'error',
                    'ledger_event_ids':[e['event_id'] for e in self.session.events()[before:]],
                    'fault':{'kind':'interrupted' if interrupted else 'controller_error',
                             'type':type(exc).__name__,'message':str(exc)}})
            raise
        with self.session.db:
            self.session.transcript.append('action_result', {'action_trace_id':action_id,'call_id':model_call_id,
                'operation':action.get('operation'),'wall_s':time.monotonic()-start,'status':'returned','result':result,
                'ledger_event_ids':[e['event_id'] for e in self.session.events()[before:]]})
        return result

    def _act(self, action):
        s = self.session
        s.freeze_checkpoints()
        if s.get('closed') or s.remaining() <= 0:
            raise ValueError('Session closed or exhausted')
        op = action['operation']
        if op == 'edit':
            path = self.workspace/action['path']
            if not path.resolve().is_relative_to(self.workspace) or path.is_symlink():
                raise ValueError('Path escapes workspace')
            if not isinstance(action['text'], str) or len(action['text'].encode()) > 1024*1024:
                raise ValueError('Edit exceeds 1 MiB')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(action['text'])
            with s.db:
                result = s.append({'kind': 'edit', 'path': action['path'], 'content_sha256': digest(action['text'])})
            return {'event_id': result['event_id'], 'content_sha256': result['content_sha256']}
        if op == 'command':
            event, result = s.run(self.workspace, action['argv'], action['decision'],
                                  seed=action.get('seed', 0), config={'image': self.image},
                                  backend=self.backend, image=self.image)
            return {'event_id': event['event_id'], **result}
        if op == 'evaluate':
            required = {'hypothesis', 'expected_result', 'evidence_ids', 'code_diff'}
            if not isinstance(action.get('decision'), dict) or not required <= set(action['decision']):
                raise ValueError('Substantive evaluation decision required')
            known_ids = {e['event_id'] for e in s.events()}
            decision = action['decision']
            if (not isinstance(decision['evidence_ids'], list)
                    or any(e not in known_ids for e in decision['evidence_ids'])
                    or any(not isinstance(decision[k], str) or not decision[k].strip()
                           for k in required-{'evidence_ids'})):
                raise ValueError('Invalid evaluation decision or evidence')
            # Consume before validating artifacts: failed evaluations count.
            with s.db:
                s.db.execute('UPDATE settings SET value=value WHERE key="used"')
                if s.get('used') >= s.get('development_calls'):
                    raise ValueError('Development budget exhausted')
                s.put('used', s.get('used')+1)
                event = s.append({'kind': 'evaluation', 'decision': action['decision']})
            try:
                source = self.workspace/'artifact'
                hashes = tree_hashes(source)
                if not hashes:
                    raise ValueError('Missing artifact')
                candidate = digest(hashes)
                frozen = s.root/'candidates'/candidate
                if not frozen.exists():
                    shutil.copytree(source, frozen)
                result, _ = evaluate(frozen, self.development_manifest, backend=self.backend,
                                     image=self.image, wall_s=s.remaining(), on_tick=s.freeze_checkpoints)
                with s.db:
                    s.append({'kind': 'candidate', 'candidate_id': candidate, 'hashes': hashes,
                              'evaluation_id': event['event_id']})
                result['candidate_id'] = candidate
            except Exception as exc:
                result = invalid_score(f'{type(exc).__name__}: {exc}')
            with s.db:
                s.append({'kind': 'development_result', 'experiment_id': event['event_id'], 'result': result})
            return result
        if op in ('select', 'submit'):
            return s.select(action['candidate_id'], action['evidence_ids'], action['reason'], op == 'submit')
        raise ValueError('Unknown operation')

    def run(self, provider):
        while not self.observe()['closed'] and self.session.remaining() > 0:
            observation = self.observe()
            with self.session.db:
                self.session.transcript.append('provider_observation', {'observation':observation,'source':'caller_provider'})
            self.act(provider(observation))
        return self.observe()


def stdio(environment):
    import json
    def emit(value):
        with environment.session.db:
            environment.session.transcript.append('protocol_output',value)
        print(json.dumps(value, allow_nan=False), flush=True)
    emit({'observation': environment.observe()})
    for line in sys.stdin:
        try:
            if len(line) > 2_000_000:
                raise ValueError('Action exceeds 2 MB')
            result = environment.act(json.loads(line))
        except (ValueError, KeyError, TypeError) as exc:
            result = {'execution_valid': False, 'failure_reason': str(exc)}
        emit({'result': result, 'observation': environment.observe()})
        if environment.observe()['closed'] or environment.session.remaining() <= 0:
            break
