"""Explicitly synthetic structural cases, never physiological reference data."""
import math
from .campaign import STAGES


def campaign_episode(seed=0):
    if type(seed) is not int or not 0 <= seed < 100:
        raise ValueError('seed must be an integer in 0..99')
    # Generator parameters stay in the trusted fixture builder, outside observations.
    shift = (seed % 3) * 5
    definitions = [
        ('short', 90, 0, 0, 40+shift, 180+shift, 1.0),
        ('long', 600, 0, 0, 40+shift, 180+shift, .5),
        ('loaded', 600, 15, 0, 60+shift, 260+shift, .5),
        ('later', 600, 15, 7, 90+shift, 260+shift, .25),
    ]
    sessions = []
    for stage, (condition, duration, load, day, fast, slow, mix) in enumerate(definitions):
        roles = [('support', 'calibration')] if stage == 0 else [('acquisition', 'diagnostic')]
        roles += [('development', 'feedback'), ('test', 'hidden')]
        for role, label in roles:
            baseline = 65 + ((stage+seed) % 3)
            start = 135 + {'calibration':0, 'diagnostic':5, 'feedback':9, 'hidden':16}[label]
            context = {'bout_duration_s': duration, 'load_kg': load, 'session_day': day}
            rows = [{'time_s': -60, 'hr_bpm': baseline, **context},
                    {'time_s': 0, 'hr_bpm': start, **context}]
            for t in (30, 60, 120, 180, 240, 360):
                y = baseline + (start-baseline)*(mix*math.exp(-t/fast)+(1-mix)*math.exp(-t/slow))
                rows.append({'time_s': t, 'hr_bpm': y, **context})
            sessions.append({'id': f'{condition}-{label}', 'participant_id': f'synthetic-{seed}',
                             'role': role, 'condition': condition, 'release_stage': stage,
                             'cost': 1, 'description': f'{duration}s bout, {load}kg recorded load, session day {day}; complete recovery recording',
                             'rows': rows})
        # An available but redundant short-bout acquisition competes with diagnostic
        # evidence under a shared budget; the menu never fabricates experiments.
        if stage > 0:
            extra = dict(sessions[0])
            extra.update(id=f'repeat-short-{stage}', role='acquisition', release_stage=stage,
                         description='Additional short unloaded day-0 recovery; same condition as initial calibration')
            sessions.append(extra)
    return {'schema_version': 1, 'id': f'structural-recovery-v1-{seed}', 'origin': 'synthetic',
            'budgets': {'acquisition': 3, 'evaluations': 8}, 'sessions': sessions,
            'campaign': {'stages': list(STAGES),
                         'context_fields': {'bout_duration_s': 'recorded prior activity duration',
                                            'load_kg': 'recorded carried load', 'session_day': 'recorded session day'},
                         'briefs': [
                             'Build a personalized executable recovery model from sparse short-bout calibration. Rest follows cutoff zero; the -60s HR sample is a recorded pre-activity rest baseline.',
                             'Investigate recovery after longer activity. Distinguish parameter error from missing structure using permitted evidence. Check earlier short-bout performance.',
                             'Transfer to a recorded carried-load condition. Choose useful calibration; evaluate both new and earlier conditions.',
                             'Sustain across later sessions. Preserve earlier-condition skill while adapting and select a final artifact from your research history.']}}
