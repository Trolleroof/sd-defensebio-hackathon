"""Linked research rounds sharing one workspace and one resource ledger."""
import copy
from pathlib import Path
from .workspace import Workspace, read, write


STAGES = ('Build', 'Investigate', 'Transfer', 'Sustain')


def validate_campaign(ep):
    from .workspace import validate_episode
    validate_episode(ep)
    if ep.get('campaign', {}).get('stages') != list(STAGES):
        raise ValueError('campaign must declare Build, Investigate, Transfer, Sustain')
    for session in ep['sessions']:
        if type(session.get('release_stage')) is not int or not 0 <= session['release_stage'] < 4:
            raise ValueError('each session requires a release stage in 0..3')
        if not isinstance(session.get('condition'), str) or not session['condition']:
            raise ValueError('each session requires a condition label')
    for stage in range(4):
        for role in ('development', 'test'):
            if not any(s['release_stage'] == stage and s['role'] == role for s in ep['sessions']):
                raise ValueError('each stage requires independent development and test sessions')
    if not any(s['role'] == 'support' and s['release_stage'] == 0 for s in ep['sessions']):
        raise ValueError('initial calibration required')


class CampaignWorkspace(Workspace):
    @classmethod
    def create(cls, episode, control_dir, workspace):
        validate_campaign(episode)
        # Base creation exports all sessions; replace with bounded initial views
        # before returning any observation to a participant.
        obj = super().create(episode, control_dir, workspace)
        obj.state.update(stage=0, checkpoints=[])
        obj._save()
        obj._publish()
        return obj

    def _stage(self):
        return read(self.root/'state.json').get('stage', 0)

    def _support(self):
        stage = self._stage()
        state = read(self.root/'state.json')
        return [copy.deepcopy(s) for s in self.ep['sessions']
                if s['release_stage'] <= stage and
                (s['role'] == 'support' or s['id'] in state['acquired'])]

    @staticmethod
    def _request(s):
        return {'id': s['id'], 'role': s['role'], 'condition': s['condition'],
                'history': [r for r in s['rows'] if r['time_s'] <= 0],
                'forecast_times_s': [r['time_s'] for r in s['rows'] if r['time_s'] > 0]}

    def public_observation(self):
        stage = self._stage()
        state = read(self.root/'state.json')
        released = [s for s in self.ep['sessions'] if s['release_stage'] <= stage]
        brief = {'schema_version': 1, 'episode_id': self.ep['id'], 'origin': self.ep['origin'],
                 'stage': STAGES[stage], 'stage_index': stage, 'stage_count': 4,
                 'task': self.ep['campaign']['briefs'][stage],
                 'interval_level': .9, 'budgets': self.ep['budgets'],
                 'context_fields': self.ep['campaign']['context_fields'],
                 'acquisition_menu': [{'id': s['id'], 'cost': s['cost'], 'description': s['description']}
                                      for s in released if s['role'] == 'acquisition' and s['id'] not in state['acquired']],
                 'checkpoint_rule': 'Checkpoint locks this round and releases the next. Final submit is available only in Sustain. Private checkpoint scores never return as feedback.'}
        return {'brief': brief, 'recordings': self._support(),
                'requests': [self._request(s) for s in released if s['role'] == 'development']}

    def _publish(self):
        observation = self.public_observation()
        write(self.visible/'brief.json', observation['brief'])
        write(self.visible/'support.json', observation['recordings'])
        write(self.visible/'requests.json', observation['requests'])

    def _begin(self, action, decision):
        event = super()._begin(action, decision)
        event['stage'] = self._stage()
        # Snapshot exactly the evidence available at the decision, not the final
        # contents of mutable public workspace files.
        event['public_snapshot'] = self.public_observation()
        self._save()
        return event

    def acquire(self, session_id, decision):
        stage = self._stage()
        session = next((s for s in self.ep['sessions'] if s['id'] == session_id), None)
        if not session or session['release_stage'] > stage:
            raise ValueError('recording has not been released')
        result = super().acquire(session_id, decision)
        self._publish()
        return result

    def _execute(self, code, role, executor):
        stage = self._stage()
        sessions = [s for s in self.ep['sessions'] if s['role'] == role and
                    (role == 'test' or s['release_stage'] <= stage)]
        requests = [self._request(s) for s in sessions]
        from .executor import execute
        if executor is execute:
            return executor(code, self._support(), requests, forbidden_paths=[str(self.root)])
        return executor(code, self._support(), requests)

    def _score_role(self, forecasts, role):
        from .workspace import score
        stage = self._stage()
        sessions = [s for s in self.ep['sessions'] if s['role'] == role and
                    (role == 'test' or s['release_stage'] <= stage)]
        if set(forecasts) != {s['id'] for s in sessions}:
            raise ValueError('exact released-session coverage required')
        scores = {s['id']: score(s['rows'], forecasts[s['id']]) for s in sessions}
        grouped = {}
        for s in sessions:
            grouped.setdefault(s['condition'], []).append(scores[s['id']])
        conditions = {name: {metric: sum(row[metric] for row in rows)/len(rows)
                              for metric in ('mae_bpm', 'interval_score_90', 'coverage_90')}
                      for name, rows in grouped.items()}
        return {'valid': True, 'sessions': scores, 'conditions': conditions,
                **{metric: sum(row[metric] for row in conditions.values())/len(conditions)
                   for metric in ('mae_bpm', 'interval_score_90', 'coverage_90')}}

    def checkpoint_code(self, code, decision, executor=None):
        from .executor import execute
        if self._stage() >= 3:
            raise ValueError('Sustain ends with final submit, not checkpoint')
        event = self._begin('checkpoint', decision)
        stage = self.state['stage']
        digest = self._freeze_candidate(code, event)
        candidate = self._load_candidate(digest)
        try:
            executed = self._execute(candidate, 'test', executor or execute)
            result = (self._score_role(executed['predictions'], 'test')
                      if executed['execution']['valid'] else {'valid': False})
            result['execution'] = executed['execution']
            result['code_execution_verified'] = bool(result['valid'])
        except Exception as exc:
            result = {'valid': False, 'error': str(exc), 'code_execution_verified': False}
        # Future-condition test scores measure forward generalization privately;
        # all fitting uses ONLY support available when this checkpoint was taken.
        write(self.root/f'checkpoint-{stage}.json', {'stage': STAGES[stage],
              'artifact_sha256': digest, 'support_ids': [s['id'] for s in self._support()], 'score': result})
        self.state['checkpoints'].append({'stage': stage, 'artifact_sha256': digest})
        self.state['stage'] += 1
        receipt = self._finish(event, {'checkpointed': STAGES[stage], 'artifact_sha256': digest,
                                      'next_stage': STAGES[stage+1]})
        self._publish()
        return receipt

    def submit_code(self, code, decision, executor=None):
        if self._stage() != 3:
            raise ValueError('complete the earlier rounds before final submission')
        return super().submit_code(code, decision, executor)

    def evaluate(self, forecasts, decision):
        raise ValueError('campaigns require executable predictors')

    def submit(self, code, forecasts, decision):
        raise ValueError('campaigns require executable predictors')


def open_workspace(control_dir):
    ep = read(Path(control_dir)/'episode.json')
    return CampaignWorkspace(control_dir) if 'campaign' in ep else Workspace(control_dir)
