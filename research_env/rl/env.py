"""Gym-style multi-turn environment over the shared ResearchEnvironment controller.

Layout per episode (all outside the participant workspace except `workspace/`):
    run_root/episodes/<task_id>-s<seed>-<n>/{workspace/, controller/, private/}
Private feedback manifests are written at reset; reward manifests only at termination.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from research_env.core.agent import ResearchEnvironment
from research_env.rl.api import PROFILES, TaskSpec, get_family, validate_against_splits, validate_spec
from research_env.rl.reward import REWARD_VERSION, compute_reward

OPERATIONS = ('edit', 'command', 'evaluate', 'select', 'submit')
_DECISION = {'type': 'object', 'description': 'Substantive research decision recorded in the ledger',
             'properties': {'hypothesis': {'type': 'string'}, 'expected_result': {'type': 'string'},
                            'evidence_ids': {'type': 'array', 'items': {'type': 'string'},
                                             'description': 'Existing ledger event IDs this decision relies on'},
                            'code_diff': {'type': 'string', 'description': 'Summary of the code change tested'}},
             'required': ['hypothesis', 'expected_result', 'evidence_ids', 'code_diff']}
_SELECTION = {'candidate_id': {'type': 'string'}, 'reason': {'type': 'string'},
              'evidence_ids': {'type': 'array', 'items': {'type': 'string'}}}
TOOL_SPECS = {
    'edit': ('Write a UTF-8 text file (<=1 MiB) inside the workspace.',
             {'path': {'type': 'string', 'description': 'Workspace-relative path'}, 'text': {'type': 'string'}},
             ['path', 'text']),
    'command': ('Run a bounded, offline command (argv list) in the workspace; consumes wall time.',
                {'argv': {'type': 'array', 'items': {'type': 'string'}}, 'decision': _DECISION,
                 'seed': {'type': 'integer'}}, ['argv', 'decision']),
    'evaluate': ('Freeze workspace artifact/ and score it on development feedback people; consumes one development call.',
                 {'decision': {'type': 'object'}}, ['decision']),
    'select': ('Declare the incumbent candidate (frozen at checkpoints) without ending the episode.',
               _SELECTION, list(_SELECTION)),
    'submit': ('Submit the final candidate and end the episode.', _SELECTION, list(_SELECTION)),
}
_REQUIRED_TYPES = {'path': str, 'text': str, 'argv': list, 'decision': dict, 'candidate_id': str,
                   'reason': str, 'evidence_ids': list, 'seed': int}
SYSTEM_PROMPT = ('You are a physiology research agent working in an offline, CPU-only workspace. '
                 'Use the tools to edit code, run bounded experiments, evaluate a candidate artifact on '
                 'development feedback people, and finally submit one evaluated candidate. Budgets: '
                 '{budgets}. The reward is computed once, after submission, on hidden people.')


class PhysiologyResearchEnv:
    def __init__(self, prepared_roots, run_root, *, backend='docker', image=None,
                 reward_version=REWARD_VERSION, max_turns=40, history_chars=60_000, profiles=None):
        if reward_version != REWARD_VERSION:
            raise ValueError(f'Unsupported reward version {reward_version!r}')
        self.prepared_roots = {k: Path(v).resolve() for k, v in prepared_roots.items()}
        self.run_root = Path(run_root).resolve()
        self.backend, self.image, self.reward_version = backend, image, reward_version
        self.max_turns, self.history_chars = int(max_turns), int(history_chars)
        # Environment-wide budget overrides (e.g. tests); never per episode.
        self.profiles = copy.deepcopy(PROFILES)
        for name, values in (profiles or {}).items():
            self.profiles[name] = {**self.profiles[name], **values}
        for name, p in self.profiles.items():
            if not 0 < p['wall_s'] <= 7200 or not 0 <= p['development_calls'] <= 30:
                raise ValueError(f'Profile {name} budgets outside 0<wall_s<=7200, 0<=calls<=30')
        self.environment = None
        self._episodes = 0

    # ---- lifecycle -------------------------------------------------------
    def reset(self, task: TaskSpec, seed: int):
        self.close()
        validate_spec(task)
        family = get_family(task.task_family)
        if task.task_family not in self.prepared_roots:
            raise ValueError(f'No prepared root for {task.task_family}')
        root = self.prepared_roots[task.task_family]
        if hasattr(family, 'dataset_splits'):
            validate_against_splits(task, family.dataset_splits(root))
        self.task, self.seed, self.family, self.prepared_root = task, int(seed), family, root
        profile = self.profiles[task.profile]
        self.budgets = {'max_turns': self.max_turns, 'wall_s': profile['wall_s'],
                        'development_calls': profile['development_calls'], 'replay': profile['replay'],
                        'profile': task.profile}
        self.episode_dir = self.run_root/'episodes'/f'{task.task_id}-s{self.seed}-{self._episodes:04}'
        self._episodes += 1
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        workspace, controller, private = (self.episode_dir/k for k in ('workspace', 'controller', 'private'))
        workspace.mkdir()
        family.stage_workspace(task, root, workspace)
        feedback = family.private_manifest(task, root, 'feedback', private)
        if Path(feedback).resolve().is_relative_to(workspace.resolve()):
            raise ValueError('Private manifest inside workspace')

        def scorer(artifact, manifest, *, backend, image, wall_s, on_tick=None):
            return family.score(artifact, manifest, backend=backend, image=image, wall_s=wall_s,
                                profile=task.profile)
        self.environment = ResearchEnvironment(
            controller, workspace, feedback, backend=self.backend, image=self.image,
            wall_s=profile['wall_s'], development_calls=profile['development_calls'], scorer=scorer,
            task_path=getattr(family, 'task_path', f'research_env/tasks/{family.name}/TASK.md'))
        self.turns, self.invalid_actions, self.done, self.done_reason = 0, 0, False, None
        self.reward_info = None
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT.format(budgets=json.dumps(self.budgets)),
                          'tool_calls': [], 'tool_call_id': None}]
        obs = self._observation(None, None)
        self.messages.append({'role': 'user', 'content': self.render_observation(obs),
                              'tool_calls': [], 'tool_call_id': None})
        return obs

    def close(self):
        if self.environment is not None:
            self.environment.session.db.close()
            self.environment = None

    # ---- stepping --------------------------------------------------------
    def step(self, action, *, assistant_message=None):
        if self.environment is None:
            raise RuntimeError('Call reset() before step()')
        if self.done:
            obs = self._observation(None, 'Episode already finished')
            return obs, 0.0, True, self._info()
        self.turns += 1
        self._record_assistant(action, assistant_message)
        error, result = None, None
        if self.environment.session.remaining() <= 0:
            error = 'Wall budget exhausted'
        else:
            try:
                self._check(action)
                result = self.environment.act(action)
            except Exception as exc:  # errors become observations, never raise to the trainer
                error = f'{type(exc).__name__}: {exc}'
        if error is not None and error != 'Wall budget exhausted':
            self.invalid_actions += 1
        session = self.environment.session
        if session.get('closed'):
            self.done, self.done_reason = True, 'submitted'
        elif session.remaining() <= 0:
            self.done, self.done_reason = True, 'wall_budget_exhausted'
        elif self.turns >= self.max_turns:
            self.done, self.done_reason = True, 'max_turns'
        obs = self._observation(result, error)
        self.messages.append({'role': 'tool', 'content': self._render_result(result, error),
                              'tool_calls': [], 'tool_call_id': self._last_call_id})
        reward = self._terminal_reward() if self.done else 0.0
        return obs, reward, self.done, self._info()

    def terminate(self, reason):
        """End the episode without a further action (e.g. policy context exhausted); scores as unsubmitted."""
        if self.environment is None:
            raise RuntimeError('Call reset() before terminate()')
        if self.done:
            return self._observation(None, 'Episode already finished'), 0.0, True, self._info()
        self.done, self.done_reason = True, f'terminated: {reason}'
        reward = self._terminal_reward()
        return self._observation(None, None), reward, True, self._info()

    def _check(self, action):
        if not isinstance(action, dict) or action.get('operation') not in OPERATIONS:
            raise ValueError(f'Action must be a dict with operation in {OPERATIONS}')
        _, properties, required = TOOL_SPECS[action['operation']]
        unknown = set(action)-{'operation', *properties}
        if unknown:
            raise ValueError(f'Unknown action fields {sorted(unknown)}')
        for key in required:
            if key not in action:
                raise ValueError(f'Missing required field {key!r}')
        for key, value in action.items():
            expected = _REQUIRED_TYPES.get(key)
            if expected and (not isinstance(value, expected) or (expected is int and isinstance(value, bool))):
                raise ValueError(f'Field {key!r} must be {expected.__name__}')

    def _terminal_reward(self):
        s = self.environment.session
        events = s.events()
        final = [e for e in events if e['kind'] == 'final_selection']
        any_valid = any(e['kind'] == 'development_result' and e['result'].get('execution_valid') for e in events)
        reward_result, baseline = None, None
        if final:
            candidate = s.root/'candidates'/final[-1]['candidate_id']
            manifest = self.family.private_manifest(self.task, self.prepared_root, 'reward',
                                                    self.episode_dir/'private')
            reward_result = self.family.score(candidate, manifest, backend=self.backend, image=self.image,
                                              wall_s=self.budgets['wall_s'], profile=self.task.profile)
            if reward_result.get('execution_valid'):
                baseline = self.family.baseline_loss(self.task, self.prepared_root)
        reward, components = compute_reward(
            submitted=bool(final), reward_result=reward_result, baseline_loss=baseline,
            any_valid_evaluation=any_valid, invalid_actions=self.invalid_actions, turns=self.turns,
            development_calls_used=s.get('used'), reward_version=self.reward_version)
        self.reward_info = {'reward': reward, 'components': components, 'reward_result': reward_result}
        return reward

    # ---- observations ----------------------------------------------------
    def _observation(self, result, error):
        base = self.environment.observe()
        return {'task': {'task_family': self.task.task_family, 'task_id': self.task.task_id,
                         'outer_split': self.task.outer_split, 'profile': self.task.profile,
                         'primary_unit': self.family.primary_unit,
                         'n_fit_people': len(self.task.fit_people),
                         'n_feedback_people': len(self.task.feedback_people)},
                'turn': self.turns, 'max_turns': self.max_turns, 'budgets': self.budgets,
                'remaining_wall_s': base['remaining_wall_s'],
                'development_calls_remaining': base['development_calls_remaining'],
                'selected_candidate_id': base['selected_candidate_id'], 'closed': base['closed'],
                'done': self.done, 'last_result': result, 'error': error,
                'history': base['history'], 'protocol': base['protocol']}

    def _render_result(self, result, error, limit=8000):
        if error is not None:
            return json.dumps({'error': error})
        result = dict(result or {})
        for key in ('stdout', 'stderr'):
            if isinstance(result.get(key), str) and len(result[key]) > 2000:
                result[key] = '...[truncated]\n'+result[key][-2000:]
        text = json.dumps(result, default=str, sort_keys=True)
        return text if len(text) <= limit else text[:limit]+'...[truncated]'

    def render_observation(self, obs):
        head = {k: obs[k] for k in ('task', 'turn', 'max_turns', 'remaining_wall_s',
                                    'development_calls_remaining', 'selected_candidate_id', 'closed', 'done')}
        parts = ['# Observation', json.dumps(head, sort_keys=True)]
        if obs.get('error'):
            parts += ['## Error', obs['error']]
        if obs.get('last_result') is not None:
            parts += ['## Last result', self._render_result(obs['last_result'], None)]
        history = [json.dumps(e, default=str, sort_keys=True) for e in obs.get('history', [])]
        budget = max(0, self.history_chars-sum(len(p) for p in parts))
        kept, used = [], 0
        for line in reversed(history):  # keep the most recent events
            if used+len(line)+1 > budget:
                break
            kept.append(line)
            used += len(line)+1
        dropped = len(history)-len(kept)
        parts.append(f'## Ledger history ({len(kept)} most recent of {len(history)} events'
                     + (f'; {dropped} older truncated' if dropped else '') + ')')
        parts.extend(reversed(kept))
        return '\n'.join(parts)

    def tool_schemas(self):
        return [{'type': 'function', 'function': {'name': name, 'description': description,
                 'parameters': {'type': 'object', 'properties': properties, 'required': required,
                                'additionalProperties': False}}}
                for name, (description, properties, required) in TOOL_SPECS.items()]

    @staticmethod
    def action_from_tool_call(name, arguments):
        """OpenAI tool call -> action dict. Malformed JSON becomes an action that step() rejects."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return {'operation': '__malformed__'}
        return {'operation': name, **arguments} if isinstance(arguments, dict) else {'operation': '__malformed__'}

    # ---- bookkeeping -----------------------------------------------------
    def _record_assistant(self, action, assistant_message):
        call_id = f'call_{self.turns:04}'
        if assistant_message is None:
            arguments = {k: v for k, v in action.items() if k != 'operation'} if isinstance(action, dict) else {}
            name = action.get('operation') if isinstance(action, dict) else None
            assistant_message = {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': call_id, 'type': 'function',
                 'function': {'name': str(name), 'arguments': json.dumps(arguments, default=str)}}]}
        message = {'role': 'assistant', 'content': assistant_message.get('content') or '',
                   'tool_calls': list(assistant_message.get('tool_calls') or []), 'tool_call_id': None}
        if message['tool_calls']:
            call_id = message['tool_calls'][0].get('id') or call_id
        self._last_call_id = call_id
        self.messages.append(message)

    def _info(self):
        s = self.environment.session
        info = {'invalid_actions': self.invalid_actions, 'turns': self.turns,
                'development_calls_used': s.get('used'), 'budgets': dict(self.budgets),
                'done_reason': self.done_reason, 'task_id': self.task.task_id, 'seed': self.seed,
                'reward_version': self.reward_version}
        events = s.events()
        info['ledger_head_sha256'] = events[-1]['sha256'] if events else None
        if self.reward_info is not None:
            info['reward_components'] = self.reward_info['components']
        return info
