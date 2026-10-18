"""Frozen terminal reward `physiology-reward-v1`. Any numeric change requires a new version string."""
from __future__ import annotations
import math

REWARD_VERSION = 'physiology-reward-v1'
EPS = 1e-9
INVALID_SUBMISSION = -1.25
NO_SUBMISSION_VALID_EVALUATION = -1.5
NO_VALID_EVALUATION = -2.0


def compute_reward(*, submitted, reward_result, baseline_loss, any_valid_evaluation,
                   invalid_actions=0, turns=0, development_calls_used=0, reward_version=REWARD_VERSION):
    """Return (reward, components) for one finished episode.

    submitted: a final selection was made. reward_result: the family `score` of the
    submitted candidate on reward people (ignored when not submitted).
    any_valid_evaluation: at least one development `evaluate` was execution_valid.
    """
    if reward_version != REWARD_VERSION:
        raise ValueError(f'Unsupported reward version {reward_version!r}')
    loss, skill = None, None
    if submitted:
        valid = bool(reward_result and reward_result.get('execution_valid'))
        loss = reward_result.get('primary_loss') if valid else None
        if valid and (type(loss) not in (int, float) or not math.isfinite(loss)):
            valid, loss = False, None  # invalid never becomes zero loss
        if valid:
            if type(baseline_loss) not in (int, float) or not math.isfinite(baseline_loss) or baseline_loss < 0:
                raise ValueError('Frozen baseline loss must be finite and nonnegative')
            skill = 1-loss/max(baseline_loss, EPS)
            tier, reward = 'valid_submission', float(min(1., max(-1., skill)))
        else:
            tier, reward = 'invalid_submission', INVALID_SUBMISSION
    elif any_valid_evaluation:
        tier, reward = 'no_submission_valid_evaluation', NO_SUBMISSION_VALID_EVALUATION
    else:
        tier, reward = 'no_valid_evaluation', NO_VALID_EVALUATION
    components = {'tier': tier, 'loss': loss, 'baseline_loss': baseline_loss, 'eps': EPS, 'skill': skill,
                  'invalid_actions': invalid_actions, 'turns': turns,
                  'development_calls_used': development_calls_used, 'reward_version': REWARD_VERSION,
                  'failure_reason': None if not submitted or tier == 'valid_submission'
                  else (reward_result or {}).get('failure_reason') or 'invalid_reward_score'}
    return reward, components
