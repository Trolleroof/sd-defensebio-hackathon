"""Write the verl RL dataset: one row per TaskSpec (verl repeats each row rollout.n = G times).

Rows follow verl's RLHFDataset layout (see examples/data_preprocess/gsm8k_tool_agent_loop.py in
verl v0.9.1): ``data_source``, ``agent_name``, ``prompt``, ``reward_model``, ``extra_info``. The
real prompt is produced by ``env.reset`` inside the agent loop; ``prompt`` here is a short stub
used only by verl's prompt-length filter. ``agent_final_evaluation`` tasks are refused.

    python -m training.build_dataset --config training/grpo_config.yaml --out /workspace/data/rl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from training import env_client
from training.verl_overrides import AGENT_NAME


def rows_for(tasks: Iterable[Any], env_seed: int = 0, group_tag: int | str = 0) -> list[dict]:
    rows = []
    for i, task in enumerate(tasks):
        env_client.refuse_final_evaluation(task)
        spec = env_client.task_to_dict(task)
        rows.append({
            'data_source': f'physiology-rl/{spec["task_family"]}',
            'agent_name': AGENT_NAME,
            'prompt': [{'role': 'user', 'content': f'Physiology research task {spec["task_id"]} ({spec["task_family"]}).'}],
            'reward_model': {'style': 'env', 'ground_truth': ''},
            'extra_info': {'task_spec': json.dumps(spec, sort_keys=True), 'env_seed': int(env_seed),
                           'group_tag': group_tag, 'index': i},
        })
    return rows


def split_pool(pool: Sequence[Any], train_split: str, dev_split: str) -> tuple[list, list]:
    train = [t for t in pool if env_client.task_field(t, 'outer_split') == train_split]
    dev = [t for t in pool if env_client.task_field(t, 'outer_split') == dev_split]
    return train, dev


def main(argv: Sequence[str] | None = None) -> int:
    from training.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).with_name('grpo_config.yaml')))
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pool = env_client.build_pools(cfg['tasks']['families'], cfg['env']['prepared_roots'], int(cfg['tasks']['pool_seed']))
    train, dev = split_pool(pool, cfg['tasks']['outer_split'], cfg['tasks']['dev_outer_split'])
    import pandas as pd  # GPU image has pandas + pyarrow
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_for(train)).to_parquet(out / 'train.parquet')
    pd.DataFrame(rows_for(dev)).to_parquet(out / 'dev.parquet')
    print(json.dumps({'train_tasks': len(train), 'dev_tasks': len(dev), 'out': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
