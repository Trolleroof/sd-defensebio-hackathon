"""Model-independent JSON action protocol; the provider receives public observations only."""
import json
from .workspace import read, write


class AgentSession:
    def __init__(self, workspace, max_steps=24):
        if type(max_steps) is not int or not 1 <= max_steps <= 1000:
            raise ValueError('max_steps must be 1..1000')
        self.workspace = workspace
        self.max_steps = max_steps
        self.path = workspace.root / 'agent-state.json'
        if not self.path.exists():
            write(self.path, {'steps': 0, 'code': None, 'candidates': {}, 'actions': [],
                              'max_steps': max_steps, 'exhausted': False})
        self.state = read(self.path)
        self.max_steps = self.state['max_steps']
        # Reconcile a completed controller edit if a previous process stopped before
        # saving the agent-side convenience state. Budgets remain controller-owned.
        for event in read(workspace.root/'state.json')['events']:
            digest = event.get('artifact_sha256')
            if event['operation'] in ('edit', 'revert') and event['status'] == 'completed' and not event.get('interrupted') and digest:
                code = workspace._load_candidate(digest)
                self.state['candidates'][digest] = code
                self.state['code'] = code
        write(self.path, self.state)

    def observe(self):
        w = self.workspace
        controller = read(w.root / 'state.json')
        # Read evidence from trusted episode copies, never paths chosen by a model.
        if hasattr(w, 'public_observation'):
            public = w.public_observation()
            recordings, requests, brief = public['recordings'], public['requests'], public['brief']
        else:
            recordings = [s for s in w.ep['sessions']
                          if s['role'] == 'support' or s['id'] in controller['acquired']]
            requests = [{'id': s['id'], 'role': s['role'],
                         'history': [r for r in s['rows'] if r['time_s'] <= 0],
                         'forecast_times_s': [r['time_s'] for r in s['rows'] if r['time_s'] > 0]}
                        for s in w.ep['sessions'] if s['role'] in ('development', 'test')]
            brief = read(w.visible / 'brief.json')
        return {'brief': brief, 'recordings': recordings,
                'requests': requests, 'remaining': controller['remaining'],
                'history': controller['events'], 'current_code': self.state['code'],
                'candidate_ids': list(self.state['candidates']),
                'steps_remaining': max(0, self.max_steps-self.state['steps']),
                'terminated': controller['submitted'] or self.state['exhausted'],
                'action_format': {'operation': 'one of the protocol keys', 'decision':
                    {'evidence': 'source reference', 'hypothesis': 'testable explanation',
                     'action': 'proposed action', 'expected_result': 'diagnostic expectation'}},
                'protocol': {'edit': ['code', 'decision'], 'acquire': ['session_id', 'decision'],
                             'evaluate': ['decision'], 'revert': ['candidate_id', 'decision'],
                             'submit': ['decision'],
                             **({'checkpoint': ['decision']} if hasattr(w, 'checkpoint_code') else {})}}

    def act(self, action):
        self.state = read(self.path)
        if self.observe()['terminated']:
            raise ValueError('agent session terminated')
        encoded = json.dumps(action, allow_nan=False)
        if len(encoded.encode()) > 300000:
            raise ValueError('action exceeds 300 KB limit')
        if len(json.dumps(self.state).encode()) + len(encoded.encode()) > 8_000_000:
            self.state['exhausted'] = True
            write(self.path, self.state)
            raise ValueError('agent history exceeds 8 MB limit')
        # Count every attempted action, including malformed/rejected calls.
        self.state['steps'] += 1
        record = {'index': self.state['steps']-1, 'action': action}
        self.state['actions'].append(record)
        write(self.path, self.state)
        try:
            if not isinstance(action, dict):
                raise ValueError('action must be an object')
            op = action['operation']
            decision = action['decision']
            w = self.workspace
            if op == 'edit':
                code = action['code']
                if not isinstance(code, str) or len(code.encode()) > 262144:
                    raise ValueError('code must be text under 256 KiB')
                event = w._begin('edit', decision)
                digest = w._freeze_candidate(code, event)
                self.state['code'] = code
                self.state['candidates'][digest] = code
                result = w._finish(event, {'candidate_id': digest})
            elif op == 'revert':
                digest = action['candidate_id']
                if digest not in self.state['candidates']:
                    raise ValueError('unknown candidate')
                event = w._begin('revert', decision)
                self.state['code'] = self.state['candidates'][digest]
                event['artifact_sha256'] = digest
                result = w._finish(event, {'candidate_id': digest})
            elif op == 'acquire':
                result = w.acquire(action['session_id'], decision)
            elif op == 'checkpoint' and hasattr(w, 'checkpoint_code'):
                if self.state['code'] is None:
                    raise ValueError('edit a predictor before checkpointing')
                result = w.checkpoint_code(self.state['code'], decision)
            elif op in ('evaluate', 'submit'):
                if self.state['code'] is None:
                    raise ValueError('edit a predictor before evaluating or submitting')
                result = (w.evaluate_code(self.state['code'], decision) if op == 'evaluate'
                          else w.submit_code(self.state['code'], decision))
            else:
                raise ValueError('unknown operation')
        except (KeyError, TypeError, ValueError) as exc:
            result = {'valid': False, 'error': str(exc)}
        record['result'] = result
        if self.state['steps'] >= self.max_steps and not read(self.workspace.root/'state.json')['submitted']:
            self.state['exhausted'] = True
            result = {**result, 'episode_status': 'step_budget_exhausted_without_submission'}
        write(self.path, self.state)
        return result

    def run(self, provider):
        """provider(observation) -> JSON action; callers own inference configuration/billing."""
        while not self.observe()['terminated']:
            self.act(provider(self.observe()))
        controller = read(self.workspace.root / 'state.json')
        return {'submitted': controller['submitted'], 'steps': self.state['steps'],
                'status': 'submitted' if controller['submitted'] else 'step_budget_exhausted'}


def stdio(session):
    """One public observation/result per line. Never emits private final scores."""
    import sys
    print(json.dumps({'observation': session.observe()}), flush=True)
    for line in sys.stdin:
        if len(line) > 1048576:
            raise ValueError('action exceeds 1 MiB')
        try:
            action = json.loads(line)
        except json.JSONDecodeError:
            action = None
        result = session.act(action)
        observation = session.observe()
        print(json.dumps({'result': result, 'observation': observation}), flush=True)
        if observation['terminated']:
            break
