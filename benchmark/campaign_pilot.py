"""Deterministic conventional competitors exercise structural campaign decisions."""
from pathlib import Path
from .agent import AgentSession
from .campaign import CampaignWorkspace, STAGES
from .campaign_examples import campaign_episode
from .research_baselines import research_codes
from .ratings import campaign_ratings
from .workspace import read, write
from .judge import build_blind_judge_packet, save_judging_report


def rationale(evidence, hypothesis, action, expectation):
    return {'evidence': evidence, 'hypothesis': hypothesis, 'action': action, 'expected_result': expectation}


def run_one(ep, root, code, acquisition='diagnostic'):
    w = CampaignWorkspace.create(ep, root/'private', root/'agent')
    session = AgentSession(w, max_steps=48)
    session.act({'operation': 'edit', 'code': code, 'decision': rationale(
        'Initial calibration and known prediction contract.',
        'This fixed conventional model family may explain permitted recordings.',
        'Install a conventional baseline candidate.', 'Measure released-session error; code is independently fitted.')})
    feedback = []
    for stage in range(4):
        if stage > 0:
            feedback.append(session.act({'operation': 'evaluate', 'decision': rationale(
                'New round requests and previous feedback.', 'Earlier fitting may fail on the new condition.',
                'Evaluate before requesting new evidence.', 'Compare error by condition to identify transfer weakness.')}))
            menu = session.observe()['brief']['acquisition_menu']
            item = next(m for m in menu if (m['id'].endswith('-diagnostic') if acquisition=='diagnostic'
                                           else m['id'] == f'repeat-short-{stage}'))
            session.act({'operation': 'acquire', 'session_id': item['id'], 'decision': rationale(
                'Menu descriptions identify workload and session context.',
                ('Condition-matched evidence should constrain the new response.' if acquisition=='diagnostic'
                 else 'Repeating the original condition may reduce calibration error.'),
                'Request '+item['id']+'.', 'The next evaluation will test whether this acquisition supports new conditions.')})
        feedback.append(session.act({'operation': 'evaluate', 'decision': rationale(
            'Allowed support, acquisitions and earlier condition errors.',
            'Refitting can improve the new condition but may damage earlier performance.',
            'Fit the current source on permitted evidence and evaluate all released development conditions.',
            'Inspect both current-condition error and retention.')}))
        op = 'submit' if stage==3 else 'checkpoint'
        result = session.act({'operation': op, 'decision': rationale(
            'All development feedback so far; no private checkpoint scores.',
            'This fixed competitor is the candidate for the current checkpoint.',
            'Freeze the current candidate and '+('finish' if stage==3 else 'continue the research campaign')+'.',
            'Private evaluation will measure current, prior and future-condition performance.')})
        if not (result.get('submitted') or result.get('checkpointed')):
            raise RuntimeError('campaign failed to progress: '+str(result))
    state = read(w.root/'state.json')
    packet = build_blind_judge_packet(w.root/'state.json', w.visible)
    write(root/'judge-packets.json', packet)
    save_judging_report(root/'judging.json',packet,state=w.root/'state.json')
    matrix = [read(w.root/f'checkpoint-{i}.json') for i in range(3)]
    final = read(w.root/'final-score.json')
    matrix.append({'stage':'Sustain','score':final})
    retention = []
    for i in range(1,4):
        prior_conditions = [s['condition'] for s in ep['sessions']
                            if s['role']=='test' and s['release_stage'] < i]
        now = matrix[i]['score']
        earlier = matrix[i-1]['score']
        deltas = ({c: now['conditions'][c]['mae_bpm']-earlier['conditions'][c]['mae_bpm'] for c in prior_conditions}
                  if now['valid'] and earlier['valid'] else None)
        retention.append({'stage':STAGES[i], 'prior_condition_error_change_bpm':deltas,
                          'meaning':'positive means regression from the previous checkpoint'})
    return {'checkpoints': matrix, 'retention':retention, 'final':final,
            'remaining':state['remaining'], 'steps':read(session.path)['steps'],
            'development_history':feedback, 'judge_status':'pending external verdicts'}


def run_campaign_pilot(output, cases=3):
    if type(cases) is not int or not 1 <= cases <= 10:
        raise ValueError('cases must be in 1..10')
    root = Path(output)
    if root.exists():
        raise ValueError('use a new output directory')
    root.mkdir(parents=True)
    runs=[]
    competitors={**{name:(code,'diagnostic') for name,code in research_codes().items()},
                 'contextual_redundant_acquisition':(research_codes()['contextual_structure_search'],'redundant')}
    for case in range(cases):
        for name,(code,policy) in competitors.items():
            run=run_one(campaign_episode(case),root/f'case-{case}'/name,code,policy)
            runs.append({'case':case,'competitor':name,**run})
    summary=[]
    for name in competitors:
        group=[r for r in runs if r['competitor']==name]
        valid=[r for r in group if r['final']['valid']]
        summary.append({'competitor':name,'completed':len(valid),'cases':cases,
                        'final_mae_bpm':sum(r['final']['mae_bpm'] for r in valid)/len(valid) if valid else None,
                        'final_interval_score_90':sum(r['final']['interval_score_90'] for r in valid)/len(valid) if valid else None})
    report={'origin':'synthetic','version':'linked-campaign-v1', 'summary':summary,'runs':runs,
            'claim':'Structural software pilot with fixed conventional competitors. No LLM result, scientific validity or unsaturated-benchmark claim.'}
    report['ratings'] = campaign_ratings(runs)
    write(root/'ratings.json', report['ratings'])
    write(root/'report.json',report)
    lines=['# Linked research campaign pilot','',report['claim'],'',
           '| Competitor | Completed | Final MAE (bpm) | Interval score |','| --- | --- | --- | --- |']
    for row in summary:
        lines.append(f"| {row['competitor']} | {row['completed']}/{cases} | {row['final_mae_bpm']} | {row['final_interval_score_90']} |")
    lines += ['', 'Private per-condition checkpoint matrices and retention deltas are in report.json.',
              'All competitors receive the same initial data, menu, 3 acquisition credits, 8 development calls and checkpoint schedule.',
              'Only the acquisition-policy ablation chooses redundant calibration. No hidden score selects an action.']
    lines += ['', '## Prediction Elo (provisional)', '',
              '| Rank | Competitor | Elo | W / D / L |', '| --- | --- | --- | --- |']
    for entry in report['ratings']['prediction']['entries']:
        lines.append(f"| {entry['rank']} | {entry['competitor']} | {entry['elo']:.1f} | {entry['wins']} / {entry['draws']} / {entry['losses']} |")
    lines += ['', 'Same synthetic family and small case batch: ratings are descriptive, not evidence of real-world superiority.',
              'Raw errors remain above; uncertainty Elo and all pairwise outcomes are in ratings.json.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return report
