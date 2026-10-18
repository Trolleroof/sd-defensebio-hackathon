"""Trusted local controller. Workspace separation is not an OS sandbox."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from .contract import predictor_contract, agent_data


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.write-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_episode(ep):
    if ep['schema_version'] != 1 or ep['origin'] not in {'synthetic', 'public', 'local'}:
        raise ValueError('unsupported episode version or origin')
    if ep['origin'] != 'synthetic' and not ep.get('provenance', {}).get('target_provenance'):
        raise ValueError('real recordings require target provenance')
    if len(json.dumps(ep, allow_nan=False).encode()) > 8_000_000:
        raise ValueError('episode exceeds 8 MB limit')
    sessions = ep['sessions']
    if not 1 <= len(sessions) <= 256 or sum(len(s['rows']) for s in sessions) > 100000:
        raise ValueError('episode session/row limit exceeded')
    ids = [s['id'] for s in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError('session IDs must be unique')
    if len({s['participant_id'] for s in sessions}) != 1:
        raise ValueError('v0 episodes require exactly one participant')
    if set(ep['budgets']) != {'acquisition', 'evaluations'}:
        raise ValueError('exact acquisition and evaluations budgets required')
    roles = {'support', 'acquisition', 'development', 'test'}
    if not {'support', 'development', 'test'} <= {s['role'] for s in sessions}:
        raise ValueError('support, development and test sessions required')
    for s in sessions:
        if s['role'] not in roles:
            raise ValueError('unknown role')
        rows = s['rows']
        times = [r['time_s'] for r in rows]
        if not rows or any(type(r[k]) not in (int, float) or not math.isfinite(r[k]) for r in rows for k in ('time_s', 'hr_bpm')):
            raise ValueError('finite, nonempty recordings required')
        if times != sorted(set(times)) or 0 not in times or not any(t > 0 for t in times):
            raise ValueError('ordered unique timestamps, cutoff zero and forecast targets required')
        if s['role'] == 'acquisition' and (type(s['cost']) is not int or s['cost'] < 1):
            raise ValueError('positive integer acquisition cost required')
    for value in ep['budgets'].values():
        if type(value) is not int or value < 0:
            raise ValueError('nonnegative integer budgets required')


def score(rows, forecasts):
    target = [r for r in rows if r['time_s'] > 0]
    if len(forecasts) != len(target):
        raise ValueError('all forecast times required')
    errors, intervals, covered = [], [], []
    for truth, pred in zip(target, forecasts):
        if set(pred) != {'time_s', 'mean', 'lower', 'upper'}:
            raise ValueError('unexpected forecast fields')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in pred.values()):
            raise ValueError('finite numeric predictions required')
        if pred['time_s'] != truth['time_s'] or not pred['lower'] <= pred['mean'] <= pred['upper']:
            raise ValueError('invalid forecast time or interval')
        y, lo, hi = truth['hr_bpm'], pred['lower'], pred['upper']
        errors.append(abs(y - pred['mean']))
        intervals.append(hi - lo + 20 * max(lo - y, 0) + 20 * max(y - hi, 0))
        covered.append(lo <= y <= hi)
    n = len(target)
    if not n:
        raise ValueError('no scorable targets')
    return {'mae_bpm': sum(errors)/n, 'interval_score_90': sum(intervals)/n,
            'coverage_90': sum(covered)/n, 'forecast_count': n}


class Workspace:
    """Single-controller local prototype; agent must never receive control_dir."""
    def __init__(self, control_dir):
        self.root = Path(control_dir).resolve()
        self.state = read(self.root / 'state.json')
        self.ep = read(self.root / 'episode.json')
        self.visible = Path(self.state['workspace'])
        # A restarted single controller must expose interrupted operations as failures.
        interrupted = False
        for event in self.state['events']:
            if event['status'] == 'started':
                event.update(status='completed', interrupted=True,
                             result={'valid': False, 'error': 'controller interrupted before completion'})
                interrupted = True
        if self.state['submitted'] and not (self.root/'final-score.json').exists():
            write(self.root/'final-score.json', {'valid': False, 'code_execution_verified': False,
                                               'error': 'submission interrupted before final scoring'})
        if interrupted:
            self._save()
            write(self.visible/'history.json', agent_data(self.state['events']))

    @classmethod
    def create(cls, episode, control_dir, workspace):
        validate_episode(episode)
        root, visible = Path(control_dir).resolve(), Path(workspace).resolve()
        if root == visible or root in visible.parents or visible in root.parents:
            raise ValueError('control and agent directories must be disjoint')
        if root.exists() or visible.exists():
            raise ValueError('directories must be new')
        root.mkdir(parents=True)
        visible.mkdir(parents=True)
        write(root / 'episode.json', episode)
        write(root / 'state.json', {'workspace': str(visible), 'remaining': episode['budgets'].copy(),
                                   'acquired': [], 'submitted': False, 'events': []})
        obj = cls(root)
        public = {'schema_version': 1, 'episode_id': episode['id'], 'origin': episode['origin'],
                  'task': 'Forecast recovery after cutoff 0; future condition is rest.',
                  'interval_level': .9, 'budgets': episode['budgets'],
                  'predictor_contract': predictor_contract(),
                  'provenance': episode.get('provenance', {}),
                  'acquisition_menu': [{'id': s['id'], 'cost': s['cost'], 'description': s['description']}
                                       for s in episode['sessions'] if s['role'] == 'acquisition']}
        write(visible / 'brief.json', agent_data(public))
        write(visible / 'support.json', agent_data([s for s in episode['sessions'] if s['role'] == 'support']))
        requests = [{'id': s['id'], 'role': s['role'],
                     'history': [r for r in s['rows'] if r['time_s'] <= 0],
                     'forecast_times_s': [r['time_s'] for r in s['rows'] if r['time_s'] > 0]}
                    for s in episode['sessions'] if s['role'] in ('development', 'test')]
        write(visible / 'requests.json', agent_data(requests))
        (visible / 'predictor.py').write_text('def predict(history, forecast_times_s):\n    raise NotImplementedError("Invent your predictor here")\n')
        return obj

    def _begin(self, action, decision):
        # Reload persisted state so a reopened controller cannot reset budgets.
        self.state = read(self.root / 'state.json')
        if self.state['submitted']:
            raise ValueError('episode terminated')
        required = {'evidence', 'hypothesis', 'action', 'expected_result'}
        if not isinstance(decision, dict) or set(decision) != required or any(not isinstance(v, str) or not v.strip() or len(v) > 4096 for v in decision.values()):
            raise ValueError('complete pre-action decision required')
        event = {'index': len(self.state['events']), 'operation': action, 'decision': decision, 'status': 'started'}
        self.state['events'].append(event)
        self._save()
        return event

    def _save(self):
        write(self.root / 'state.json', self.state)

    def _finish(self, event, result):
        event['status'], event['result'] = 'completed', result
        self._save()
        write(self.visible / 'history.json', agent_data(self.state['events']))
        return result

    def acquire(self, session_id, decision):
        choices = {s['id']: s for s in self.ep['sessions'] if s['role'] == 'acquisition'}
        self.state = read(self.root / 'state.json')
        if session_id not in choices or session_id in self.state['acquired']:
            raise ValueError('unavailable or already acquired recording')
        session = choices[session_id]
        if self.state['remaining']['acquisition'] < session['cost']:
            raise ValueError('acquisition budget exhausted')
        event = self._begin('acquire', decision)
        self.state['remaining']['acquisition'] -= session['cost']
        self.state['acquired'].append(session_id)
        self._save()
        # IDs are data, never filesystem paths.
        write(self.visible / f'acquisition-{len(self.state["acquired"])}.json', agent_data(session))
        return self._finish(event, {'recording': session_id, 'remaining': self.state['remaining'].copy()})

    def evaluate(self, forecasts, decision):
        self.state = read(self.root / 'state.json')
        if self.state['remaining']['evaluations'] < 1:
            raise ValueError('evaluation budget exhausted')
        event = self._begin('evaluate', decision)
        self.state['remaining']['evaluations'] -= 1
        self._save()
        try:
            result = self._score_role(forecasts, 'development')
        except (ValueError, KeyError, TypeError) as exc:
            result = {'valid': False, 'error': str(exc)}
        return self._finish(event, result)

    def _score_role(self, forecasts, role):
        sessions = [s for s in self.ep['sessions'] if s['role'] == role]
        if set(forecasts) != {s['id'] for s in sessions}:
            raise ValueError('exact session coverage required')
        scores = {s['id']: score(s['rows'], forecasts[s['id']]) for s in sessions}
        return {'valid': True, 'sessions': scores,
                **{key: sum(s[key] for s in scores.values()) / len(scores)
                   for key in ('mae_bpm', 'interval_score_90', 'coverage_90')}}

    def submit(self, code, forecasts, decision):
        event = self._begin('submit', decision)
        self.state['submitted'] = True
        self._save()
        # Freeze code and forecasts; execution parity is not yet verified.
        (self.root / 'submission.py').write_text(code)
        write(self.root / 'submission.json', forecasts)
        digest = hashlib.sha256(code.encode()).hexdigest()
        try:
            result = self._score_role(forecasts, 'test')
        except (ValueError, KeyError, TypeError) as exc:
            result = {'valid': False, 'error': str(exc)}
        result['code_execution_verified'] = False
        write(self.root / 'final-score.json', result)
        return self._finish(event, {'submitted': True, 'code_sha256': digest})


    def _execute(self, code, role, executor):
        support = [s for s in self.ep['sessions']
                   if s['role'] == 'support' or s['id'] in self.state['acquired']]
        requests = [{'id': s['id'], 'history': [r for r in s['rows'] if r['time_s'] <= 0],
                     'forecast_times_s': [r['time_s'] for r in s['rows'] if r['time_s'] > 0]}
                    for s in self.ep['sessions'] if s['role'] == role]
        from .executor import execute
        support, requests = agent_data(support), agent_data(requests)
        if executor is execute:
            return executor(code, support, requests, forbidden_paths=[str(self.root)])
        return executor(code, support, requests)

    def _freeze_candidate(self, code, event):
        if not isinstance(code, str) or len(code.encode()) > 262144:
            raise ValueError('predictor code must be text under 256 KiB')
        digest = hashlib.sha256(code.encode()).hexdigest()
        directory = self.root / 'candidates'
        directory.mkdir(exist_ok=True)
        (directory / (digest + '.py')).write_text(code)
        event['artifact_sha256'] = digest
        self._save()
        return digest

    def _load_candidate(self, digest):
        if len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
            raise ValueError('invalid artifact digest')
        code = (self.root / 'candidates' / (digest + '.py')).read_text()
        if hashlib.sha256(code.encode()).hexdigest() != digest:
            raise ValueError('candidate artifact integrity failure')
        return code

    def evaluate_code(self, code, decision, executor=None):
        from .executor import execute
        self.state = read(self.root / 'state.json')
        if self.state['remaining']['evaluations'] < 1:
            raise ValueError('evaluation budget exhausted')
        event = self._begin('evaluate_code', decision)
        self.state['remaining']['evaluations'] -= 1
        self._save()
        try:
            digest = self._freeze_candidate(code, event)
            code = self._load_candidate(digest)
            executed = self._execute(code, 'development', executor or execute)
            result = (self._score_role(executed['predictions'], 'development')
                      if executed['execution']['valid'] else {'valid': False})
            result['execution'] = executed['execution']
            result['code_execution_verified'] = bool(result['valid'])
        except Exception as exc:
            result = {'valid': False, 'error': str(exc), 'code_execution_verified': False}
        return self._finish(event, result)

    def submit_code(self, code, decision, executor=None):
        from .executor import execute
        event = self._begin('submit_code', decision)
        self.state['submitted'] = True
        self._save()
        digest = None
        try:
            digest = self._freeze_candidate(code, event)
            code = self._load_candidate(digest)
            (self.root / 'submission.py').write_text(code)
            executed = self._execute(code, 'test', executor or execute)
            result = (self._score_role(executed['predictions'], 'test')
                      if executed['execution']['valid'] else {'valid': False})
            result['execution'] = executed['execution']
            result['code_execution_verified'] = bool(result['valid'])
            if executed['execution']['valid']:
                write(self.root / 'submission.json', executed['predictions'])
        except Exception as exc:
            result = {'valid': False, 'error': str(exc), 'code_execution_verified': False}
        write(self.root / 'final-score.json', result)
        return self._finish(event, {'submitted': True, 'code_sha256': digest})
