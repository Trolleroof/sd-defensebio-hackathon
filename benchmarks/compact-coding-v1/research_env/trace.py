"""Export a live or finished private run transcript to JSONL and readable HTML."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from research_env.core.common import canonical, read, write
from research_env.core.transcript import Transcript


def export(run_root):
    root = Path(run_root)
    controller = root/'private/controller.sqlite'
    if not controller.exists():
        raise ValueError('No controller exists for this run')
    trace = Transcript(root/'private')
    try:
        records = trace.records()
    finally:
        trace.close()
    meta = read(root/'run.json') if (root/'run.json').exists() else {}
    statuses = [r['data'] for r in records if r['kind'] == 'run_status']
    status = statuses[-1] if statuses else {'status':'running', 'fault':None, 'final_answer':None}
    status = {**status, 'transcript_records':len(records),
              'last_trace_id':records[-1]['trace_id'] if records else None}
    write(root/'run-status.json', status)
    target = root/'transcript.jsonl'
    temp = target.with_suffix('.jsonl.tmp')
    temp.write_bytes(b''.join(canonical(r)+b'\n' for r in records))
    temp.replace(target)
    title = f"{meta.get('competitor','Research run')} / {meta.get('task','task')} / seed {meta.get('seed','?')}"
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>'+escape(title)+'</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#17212b;background:#f8fafc}'
             'h1{font-size:26px}details{background:white;border:1px solid #ccd5df;border-radius:8px;margin:12px 0;padding:14px}'
             'summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px ui-monospace,monospace}'
             'small{color:#506070}a{color:#145eb4}</style>',
             '<h1>'+escape(title)+'</h1>',
             '<p><strong>Status: '+escape(status['status'])+'</strong> · '+str(len(records))+' records</p>',
             '<p>Private inspection transcript. Exact prompts, returned actions and controller results are retained. '
             'Provider credentials are never included. Hidden test scores stay in the separate receipt.</p>',
             '<p><a href="transcript.jsonl">JSONL transcript</a> · <a href="run-status.json">Run status</a></p>',
             '<details><summary>Run configuration and final status</summary><pre>'+escape(json.dumps(
                 {'configuration':meta,'status':status},indent=2,ensure_ascii=False))+'</pre></details>']
    for r in records:
        timestamp = datetime.fromtimestamp(r['timestamp'],timezone.utc).isoformat(timespec='milliseconds')
        data = r['data']
        label = r['trace_id']+' · '+r['kind']
        if data.get('call_id') is not None:
            label += ' · call '+str(data['call_id'])
        if data.get('operation'):
            label += ' · '+str(data['operation'])
        parts += ['<details><summary>'+escape(label)+'</summary><small>'+escape(timestamp)+'</small><pre>'+
                  escape(json.dumps(data,indent=2,ensure_ascii=False))+'</pre></details>']
    parts += ['</html>']
    (root/'transcript.html').write_text('\n'.join(parts))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root')
    args = parser.parse_args()
    print(json.dumps(export(args.run_root),indent=2))


if __name__ == '__main__':
    main()
