"""Developer CLI. Never expose trusted controller commands to predictor code."""
import argparse
import json
from pathlib import Path
from .workspace import Workspace, read, write
from .campaign import open_workspace


def main():
    parser = argparse.ArgumentParser(description='Physiology research benchmark')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Execute synthetic baselines and scripted revision campaign')
    demo.add_argument('output')
    real_rate = sub.add_parser('rate-physbench', help='Rate completed existing SoliderPhys-Bench model runs offline')
    real_rate.add_argument('run_directory')
    real_rate.add_argument('output')
    rate = sub.add_parser('rate', help='Build versioned Elo tables from a complete matched campaign report')
    rate.add_argument('report')
    rate.add_argument('output')
    campaign = sub.add_parser('campaign-demo', help='Run linked structural research baseline campaigns')
    campaign.add_argument('output')
    campaign.add_argument('--cases', type=int, default=3)
    new_campaign = sub.add_parser('create-campaign', help='Create linked synthetic agent workspace')
    new_campaign.add_argument('output')
    new_campaign.add_argument('--seed', type=int, default=0)
    create = sub.add_parser('create', help='Create synthetic workspace')
    create.add_argument('output')
    agent = sub.add_parser('agent', help='Public JSON-lines action/observation protocol')
    agent.add_argument('control')
    agent.add_argument('--max-steps', type=int, default=24)
    audit = sub.add_parser('audit-csv', help='Validate supplied canonical recordings')
    audit.add_argument('csv')
    audit.add_argument('--origin', choices=['public', 'local'], required=True)
    audit.add_argument('--dataset', required=True)
    audit.add_argument('--target-provenance', required=True)
    audit.add_argument('--source-url')
    real = sub.add_parser('create-csv', help='Create workspace from audited canonical recordings')
    real.add_argument('csv')
    real.add_argument('output')
    real.add_argument('--origin', choices=['public', 'local'], required=True)
    real.add_argument('--dataset', required=True)
    real.add_argument('--target-provenance', required=True)
    real.add_argument('--source-url')
    real.add_argument('--participant', required=True)
    real.add_argument('--episode-id', required=True)
    verdict = sub.add_parser('import-verdicts', help='Validate external judge verdicts and save a separate report')
    verdict.add_argument('control')
    verdict.add_argument('verdicts')
    verdict.add_argument('output')
    judge = sub.add_parser('judge-packets', help='Export chronology-bounded external judge inputs')
    judge.add_argument('control')
    judge.add_argument('output')
    args = parser.parse_args()
    if args.command == 'rate-physbench':
        from .ratings import physbench_ratings
        run = Path(args.run_directory)
        result = physbench_ratings(read(run/'results.json'), read(run/'run_meta.json'))
        write(args.output, result)
        print(json.dumps(result['entries'], indent=2))
    elif args.command == 'rate':
        from .ratings import campaign_ratings
        result = campaign_ratings(read(args.report)['runs'])
        write(args.output, result)
        print(json.dumps(result['prediction']['entries'], indent=2))
    elif args.command == 'campaign-demo':
        from .campaign_pilot import run_campaign_pilot
        report = run_campaign_pilot(args.output, cases=args.cases)
        print(json.dumps({'report': str(Path(args.output).resolve()/'report.md'),
                          'summary': report['summary']}, indent=2))
    elif args.command == 'create-campaign':
        from .campaign import CampaignWorkspace
        from .campaign_examples import campaign_episode
        root = Path(args.output)
        w = CampaignWorkspace.create(campaign_episode(args.seed), root/'private', root/'agent')
        print(json.dumps({'control': str(w.root), 'workspace': str(w.visible)}))
    elif args.command == 'demo':
        from .pilot import run_pilot
        report = run_pilot(args.output)
        print(json.dumps({'report': str(Path(args.output).resolve()/'report.md'),
                          'runs': [{'name': r['name'], 'final': r['final']} for r in report['runs']]}, indent=2))
    elif args.command == 'create':
        from .examples import episode
        root = Path(args.output)
        w = Workspace.create(episode(), root/'private', root/'agent')
        print(json.dumps({'control': str(w.root), 'workspace': str(w.visible)}))
    elif args.command == 'agent':
        from .agent import AgentSession, stdio
        stdio(AgentSession(open_workspace(args.control), max_steps=args.max_steps))
    elif args.command in ('audit-csv', 'create-csv'):
        from .data import load_csv
        data = load_csv(args.csv, origin=args.origin, dataset=args.dataset,
                        target_provenance=args.target_provenance, source_url=args.source_url)
        if args.command == 'audit-csv':
            print(json.dumps(data, indent=2))
        else:
            from .data import make_episode
            ep = make_episode(data, participant_id=args.participant, episode_id=args.episode_id)
            root = Path(args.output)
            w = Workspace.create(ep, root/'private', root/'agent')
            print(json.dumps({'control': str(w.root), 'workspace': str(w.visible)}))
    elif args.command == 'import-verdicts':
        from .judge import build_blind_judge_packet, save_judging_report
        w = open_workspace(args.control)
        packet = build_blind_judge_packet(w.root/'state.json', w.visible)
        report = save_judging_report(args.output, packet, verdicts=read(args.verdicts), state=w.root/'state.json')
        print(json.dumps(report, indent=2))
    elif args.command == 'judge-packets':
        from .judge import build_blind_judge_packet
        w = open_workspace(args.control)
        packet = build_blind_judge_packet(w.root/'state.json', w.visible)
        write(args.output, packet)
        print(json.dumps({'output': str(Path(args.output).resolve()), 'actions': len(packet['event_packets'])}))


if __name__ == '__main__':
    main()
