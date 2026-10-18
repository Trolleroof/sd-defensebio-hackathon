"""Explicit builder/controller commands; no paid call or provider auto-discovery."""
import argparse
from pathlib import Path
import sys
from research_env.core.common import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    fetch = sub.add_parser('fetch')
    fetch.add_argument('source_root')
    prep = sub.add_parser('prepare')
    prep.add_argument('source_root')
    prep.add_argument('output_root')
    prep.add_argument('--splits')
    base = sub.add_parser('baseline')
    base.add_argument('train_manifest')
    base.add_argument('output_dir')
    base.add_argument('--kind', choices=('median', 'spectral', 'motion', 'ridge'), default='median')
    base.add_argument('--seed', type=int, default=0)
    base.add_argument('--alpha', type=float, default=10)
    base.add_argument('--motion-weight', type=float, default=.5)
    base.add_argument('--calibration-manifest')
    replay = sub.add_parser('replay')
    replay.add_argument('artifact_dir')
    replay.add_argument('private_manifest')
    replay.add_argument('report')
    replay.add_argument('--backend', choices=('docker', 'beam', 'macos-development', 'trusted-builder'), default='docker')
    replay.add_argument('--image')
    run = sub.add_parser('run')
    run.add_argument('controller')
    run.add_argument('workspace')
    run.add_argument('decision_json')
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--development', action='store_true')
    run.add_argument('--backend', choices=('docker', 'beam', 'macos-development'), default='docker')
    run.add_argument('--image')
    run.add_argument('command', nargs=argparse.REMAINDER)
    select = sub.add_parser('select')
    select.add_argument('controller')
    select.add_argument('candidate_id')
    select.add_argument('reason')
    select.add_argument('--evidence', nargs='*', default=[])
    select.add_argument('--final', action='store_true')
    export = sub.add_parser('trajectory')
    export.add_argument('controller')
    export.add_argument('output')
    export.add_argument('--outer-split', required=True, choices=('agent_training', 'agent_development', 'agent_final_evaluation'))
    agent = sub.add_parser('agent')
    agent.add_argument('controller')
    agent.add_argument('workspace')
    agent.add_argument('development_manifest')
    agent.add_argument('--backend', choices=('docker', 'beam', 'macos-development'), default='docker')
    agent.add_argument('--image')
    packets = sub.add_parser('judge-export')
    packets.add_argument('controller')
    packets.add_argument('output')
    live = sub.add_parser('judge-run')
    live.add_argument('packets')
    live.add_argument('config')
    live.add_argument('output_dir')
    live.add_argument('--key-env', required=True, help='Explicit environment variable containing the authorized provider key')
    judge = sub.add_parser('judge-import')
    judge.add_argument('packets')
    judge.add_argument('report')
    judge.add_argument('--verdicts')
    args = parser.parse_args()
    if args.operation == 'agent':
        from research_env.core.agent import ResearchEnvironment, stdio
        stdio(ResearchEnvironment(args.controller, args.workspace, args.development_manifest,
                                 backend=args.backend, image=args.image))
        return
    if args.operation == 'judge-run':
        import os
        from research_env.judging.judge import OpenAICompatibleJudge, judge_packet, summarize
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=False)
        provider = OpenAICompatibleJudge(api_key=os.environ[args.key_env], **read(args.config))
        packets = read(args.packets)
        verdicts = []
        for p in packets:
            judged = judge_packet(p, provider, out/(p['event_id']+'-attempts.json'))
            if judged.get('verdict') is not None:
                verdicts.append(judged['verdict'])
        write(out/'verdicts.json', verdicts)
        result = summarize(packets, verdicts)
        write(out/'report.json', result)
    elif args.operation == 'judge-export':
        from research_env.core.session import Session
        from research_env.judging.judge import build_packets
        result = build_packets(Session(args.controller).events())
        write(args.output, result)
    elif args.operation in ('fetch', 'prepare'):
        from research_env.tasks.pulse.data import fetch, prepare
        result = fetch(args.source_root) if args.operation == 'fetch' else prepare(args.source_root, args.output_root, args.splits)
    elif args.operation == 'baseline':
        from research_env.tasks.pulse.baselines import train
        result = train(args.train_manifest, args.calibration_manifest, args.output_dir, args.seed,
                       args.kind, args.alpha, args.motion_weight)
    elif args.operation == 'replay':
        if args.backend == 'trusted-builder':
            from research_env.tasks.pulse.baselines import replay
            result, _ = replay(args.artifact_dir, args.private_manifest)
            result.update(scored_resource_isolation=False, execution_context='trusted_builder_only')
        else:
            from research_env.core.evaluation import evaluate
            result, _ = evaluate(args.artifact_dir, args.private_manifest, backend=args.backend, image=args.image)
        write(args.report, result)
    elif args.operation == 'run':
        from research_env.core.session import Session
        cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
        result = Session(args.controller).run(args.workspace, cmd, read(args.decision_json), seed=args.seed,
                                             config={'backend': args.backend, 'image': args.image},
                                             backend=args.backend, image=args.image, development=args.development)
    elif args.operation == 'select':
        from research_env.core.session import Session
        result = Session(args.controller).select(args.candidate_id, args.evidence, args.reason, args.final)
    elif args.operation == 'trajectory':
        from research_env.core.session import Session
        Session(args.controller).export(args.output, args.outer_split)
        result = {'export': args.output, 'outer_split': args.outer_split}
    else:
        from research_env.judging.judge import summarize
        result = summarize(read(args.packets), read(args.verdicts) if args.verdicts else [])
        write(args.report, result)
    print(__import__('json').dumps(result, default=str, indent=2))

if __name__ == '__main__':
    main()
