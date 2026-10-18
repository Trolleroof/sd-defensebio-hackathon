"""Reproducible synthetic smoke comparison, not a model leaderboard."""
from pathlib import Path
from .agent import AgentSession
from .baselines import baseline_codes
from .data import select_baseline
from .examples import episode, decision
from .workspace import Workspace, read, write


def run_pilot(root):
    from .judge import build_blind_judge_packet, save_judging_report
    root = Path(root)
    if root.exists():
        raise ValueError('pilot output directory must be new')
    root.mkdir(parents=True)
    codes = baseline_codes()
    runs = []
    for name, code in codes.items():
        w = Workspace.create(episode(), root/name/'private', root/name/'agent')
        # Same final accessible evidence for the fixed baselines and revision campaign.
        w.acquire('long-bout', decision('Use declared additional calibration recording'))
        development = w.evaluate_code(code, decision('Evaluate fixed numerical baseline'))
        w.submit_code(code, decision('Submit fixed numerical baseline'))
        runs.append({'name': name, 'kind': 'numerical_baseline', 'development': development,
                     'final': read(w.root/'final-score.json'), 'remaining': read(w.root/'state.json')['remaining']})
    w = Workspace.create(episode(), root/'automatic_selection'/'private', root/'automatic_selection'/'agent')
    w.acquire('long-bout', decision('Use the same calibration acquisition as other competitors'))
    candidates = {name: w.evaluate_code(code, decision('Evaluate candidate '+name))
                  for name, code in codes.items()}
    try:
        selected = select_baseline(candidates)
    except ValueError:
        selected = None
    if selected is not None:
        w.submit_code(codes[selected], decision('Select the lowest valid development-error candidate'))
    else:
        write(w.root/'final-score.json', {'valid': False, 'code_execution_verified': False,
                                        'error': 'all development candidates invalid; no selection'})
    runs.append({'name': 'automatic_selection', 'kind': 'development_only_search',
                 'selected': selected, 'development': candidates,
                 'final': read(w.root/'final-score.json'),
                 'remaining': read(w.root/'state.json')['remaining']})
    w = Workspace.create(episode(), root/'scripted_revision'/'private', root/'scripted_revision'/'agent')
    session = AgentSession(w)
    actions = [
        {'operation': 'edit', 'code': codes['persistence'], 'decision': decision('Build initial persistence candidate')},
        {'operation': 'evaluate', 'decision': decision('Measure initial development error')},
        {'operation': 'acquire', 'session_id': 'long-bout', 'decision': decision('Inspect another recovery trajectory')},
        {'operation': 'edit', 'code': codes['linear_recovery'], 'decision': decision('Fit a slope from permitted recordings')},
        {'operation': 'evaluate', 'decision': decision('Check fitted candidate against development recordings')},
        {'operation': 'submit', 'decision': decision('Select fitted candidate for untouched evaluation')},
    ]
    status = session.run(lambda observation: actions.pop(0))
    packet = build_blind_judge_packet(w.root/'state.json', w.visible)
    write(root/'judge-packets.json', packet)
    judging = save_judging_report(root/'judging.json', packet, state=w.root/'state.json')
    runs.append({'name': 'scripted_revision', 'kind': 'scripted_protocol_smoke', 'agent_status': status,
                 'final': read(w.root/'final-score.json'), 'remaining': read(w.root/'state.json')['remaining']})
    report = {'schema_version': 1, 'origin': 'synthetic', 'episode_id': episode()['id'],
              'claim': 'Software smoke comparison only. No LLM inference or physiological validation.',
              'uncertainty': 'Fixed +/-5 bpm fixture intervals; no calibration claim.',
              'runs': runs, 'judging': judging}
    write(root/'report.json', report)
    lines = ['# Synthetic benchmark smoke report', '', report['claim'], '',
             '| Competitor | Complete | MAE (bpm) | Code executed |', '| --- | --- | --- | --- |']
    for run in runs:
        final = run['final']
        lines.append(f"| {run['name']} | {final.get('valid', False)} | {final.get('mae_bpm', 'invalid')} | {final.get('code_execution_verified', False)} |")
    lines += ['', 'Research-quality judging: see judging.json. Pending external LLM verdicts are not scores.',
              'These deterministic toy cases do not measure benchmark difficulty or real-session generalization.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return report
