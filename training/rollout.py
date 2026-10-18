"""GRPO group rollouts against ``PhysiologyResearchEnv``.

One *group* = G concurrent episodes of the same TaskSpec, with identical env settings and budgets
(and, by default, the same env seed); only policy sampling differs. The policy is reached through
an OpenAI-compatible endpoint (vLLM ``vllm serve ... --enable-auto-tool-choice --tool-call-parser
<fmt>``), or, inside verl, through ``training/verl_agent_loop.py`` which reuses ``run_episode``.

Tool calls are parsed into ResearchEnvironment operation dicts. A malformed call (bad JSON,
unknown tool, missing call) is still sent to the env as an explicit malformed action so the env
returns an error observation and counts it in ``info['invalid_actions']``.

CLI (standalone data collection / evaluation; no training):
    python -m training.rollout --config training/grpo_config.yaml --tasks tasks.jsonl --out traj.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from training import env_client
from training.advantage import group_advantages
from training.trajectory_io import build_record, message, write_jsonl

SYSTEM_PROMPT = (
    'You are a research agent building a personalized physiology predictor inside a sandboxed '
    'workspace. Act only by calling exactly one of the provided tools per turn. Use `evaluate` '
    'sparingly: development calls are limited. Finish by calling `submit` with your best candidate.')


class PolicyExhausted(RuntimeError):
    """Policy cannot continue (context or response-token budget exhausted)."""


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[dict] = field(default_factory=list)  # OpenAI format: {id, type, function:{name, arguments}}
    raw: Any = None


class Policy(Protocol):
    async def step(self, messages: list[dict], tools: list[dict], *, sample_index: int) -> AssistantTurn: ...


@dataclass
class ParsedCall:
    tool_call_id: str
    action: dict
    error: str | None


def parse_tool_calls(turn: AssistantTurn, names: set[str], turn_index: int) -> list[ParsedCall]:
    """Turn the model's tool calls into env actions. Never raises."""
    if not turn.tool_calls:
        err = 'No tool call found. Respond with exactly one tool call.'
        return [ParsedCall('', env_client.malformed_action(err, turn.content), err)]
    out = []
    for k, call in enumerate(turn.tool_calls):
        call_id = (call.get('id') if isinstance(call, dict) else None) or f'call_{turn_index}_{k}'
        try:
            fn = call['function']
            name, raw_args = fn['name'], fn.get('arguments', '{}')
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            out.append(ParsedCall(call_id, env_client.action_from_tool_call(name, args, names), None))
        except Exception as exc:
            err = f'Malformed tool call: {type(exc).__name__}: {exc}'
            out.append(ParsedCall(call_id, env_client.malformed_action(err, call), err))
    return out


def to_api_messages(messages: list[dict]) -> list[dict]:
    """Contract-shaped messages -> OpenAI chat API messages (drop empty tool fields)."""
    api = []
    for m in messages:
        a = {'role': m['role'], 'content': m.get('content') or ''}
        if m['role'] == 'assistant' and m.get('tool_calls'):
            a['tool_calls'] = m['tool_calls']
        if m['role'] == 'tool':
            a['tool_call_id'] = m['tool_call_id']
        api.append(a)
    return api


class OpenAIChatPolicy:
    """Policy served by vLLM's OpenAI-compatible server (requires the ``openai`` package)."""

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None, max_tokens: int = 2048,
                 temperature: float = 1.0, top_p: float = 1.0, timeout_s: float = 600, client: Any = None):
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=base_url, api_key=api_key or 'EMPTY', timeout=timeout_s)
        self.client, self.model = client, model
        self.max_tokens, self.temperature, self.top_p = max_tokens, temperature, top_p

    async def step(self, messages: list[dict], tools: list[dict], *, sample_index: int) -> AssistantTurn:
        resp = await self.client.chat.completions.create(
            model=self.model, messages=to_api_messages(messages), tools=tools, tool_choice='auto',
            max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        choice = resp.choices[0]
        # A turn truncated by max_tokens without a tool call is treated as malformed, not exhaustion.
        calls = [{'id': c.id, 'type': 'function',
                  'function': {'name': c.function.name, 'arguments': c.function.arguments}}
                 for c in (choice.message.tool_calls or [])]
        return AssistantTurn(choice.message.content or '', calls, raw=choice.finish_reason)


@dataclass
class EpisodeResult:
    record: dict
    reward: float
    budgets: dict
    invalid_actions: int
    forced_termination: bool


def group_id_for(task: Any, seed: int, step: int | str = 0) -> str:
    key = f'{env_client.task_field(task, "task_id")}|{seed}|{step}'
    return 'grp-' + hashlib.sha256(key.encode()).hexdigest()[:16]


async def run_episode(*, env: Any, task: Any, seed: int, policy: Policy, settings: env_client.EnvSettings,
                      group_id: str, sample_index: int, system_prompt: str = SYSTEM_PROMPT,
                      model_id: str | None = None) -> EpisodeResult:
    env_client.refuse_final_evaluation(task)
    obs = await asyncio.to_thread(env_client.reset, env, task, seed)
    tools = env_client.tool_schemas(env)
    names = env_client.tool_names(tools)
    budgets = env_client.budgets_of(obs, {}, settings, task)
    messages = [message('system', system_prompt), message('user', env_client.render(env, obs))]
    reward, done, info, forced, parse_errors = 0.0, False, {}, False, 0
    for turn_index in range(settings.max_turns):
        try:
            turn = await policy.step(messages, tools, sample_index=sample_index)
        except PolicyExhausted:
            forced = True
            break
        messages.append(message('assistant', turn.content, turn.tool_calls))
        for call in parse_tool_calls(turn, names, turn_index):
            parse_errors += call.error is not None
            obs, reward, done, info = await asyncio.to_thread(env_client.step, env, call.action)
            text = env_client.render_step(env, obs)
            if call.tool_call_id:
                messages.append(message('tool', text, tool_call_id=call.tool_call_id))
            else:  # no tool call at all: feedback goes back as a user turn
                messages.append(message('user', text))
            if done:
                break
        if done:
            break
    if not done:
        forced = True
        obs, reward, done, info = await asyncio.to_thread(env_client.terminate, env, 'policy stopped',
                                                          settings.max_turns)
    if not done:
        raise RuntimeError(f'Env did not terminate episode for task {env_client.task_field(task, "task_id")}')
    components = dict(info.get('reward_components') or {})
    invalid = int(info.get('invalid_actions', components.get('invalid_actions', 0)) or 0)
    record = build_record(
        task=task, seed=seed, group_id=group_id, reward=reward,
        reward_version=components.get('reward_version', settings.reward_version),
        reward_components=components, messages=messages,
        ledger_head_sha256=env_client.ledger_head(obs, info), budgets=budgets,
        rollout={'sample_index': sample_index, 'model': model_id, 'forced_termination': forced,
                 'parse_errors': parse_errors})
    return EpisodeResult(record, reward, budgets, invalid, forced)


def episode_settings(settings: env_client.EnvSettings, group_id: str, sample_index: int) -> env_client.EnvSettings:
    """Same budgets for all samples; only the scratch run directory differs."""
    return replace(settings, run_root=str(Path(settings.run_root) / group_id / f'sample-{sample_index:03d}'))


async def run_group(*, task: Any, seed: int, group_size: int, policy: Policy, settings: env_client.EnvSettings,
                    env_factory: Callable[..., Any] | None = None, step: int | str = 0,
                    env_seed_policy: str = 'shared', norm_adv_by_std: bool = True, adv_epsilon: float = 1e-6,
                    semaphore: asyncio.Semaphore | None = None, model_id: str | None = None) -> list[dict]:
    env_client.refuse_final_evaluation(task)
    if group_size < 2:
        raise ValueError('group_size must be >= 2')
    gid = group_id_for(task, seed, step)
    sem = semaphore or asyncio.Semaphore(group_size)

    async def one(i: int) -> EpisodeResult:
        async with sem:
            env = env_client.make_env(episode_settings(settings, gid, i), env_factory)
            try:
                env_seed = seed if env_seed_policy == 'shared' else seed * 1000 + i
                return await run_episode(env=env, task=task, seed=env_seed, policy=policy, settings=settings,
                                         group_id=gid, sample_index=i, model_id=model_id)
            finally:
                await asyncio.to_thread(env_client.close, env)

    results = await asyncio.gather(*(one(i) for i in range(group_size)))
    first = json.dumps(results[0].budgets, sort_keys=True, default=str)
    if any(json.dumps(r.budgets, sort_keys=True, default=str) != first for r in results):
        raise RuntimeError(f'Budgets differ within group {gid}: {[r.budgets for r in results]}')
    advs = group_advantages([r.reward for r in results], norm_by_std=norm_adv_by_std, eps=adv_epsilon)
    for r, a in zip(results, advs):
        r.record['rollout']['advantage'] = a
    return [r.record for r in results]


async def run_batch(*, tasks: Sequence[Any], seeds: Sequence[int], cfg: dict, policy: Policy,
                    settings: env_client.EnvSettings, env_factory: Callable[..., Any] | None = None,
                    step: int | str = 0) -> list[dict]:
    for t in tasks:
        env_client.refuse_final_evaluation(t)
    g = cfg['grpo']
    sem = asyncio.Semaphore(int(g.get('max_concurrent_episodes', g['group_size'])))
    groups = await asyncio.gather(*(run_group(
        task=t, seed=s, group_size=int(g['group_size']), policy=policy, settings=settings, env_factory=env_factory,
        step=step, env_seed_policy=g.get('env_seed_policy', 'shared'), norm_adv_by_std=bool(g['norm_adv_by_std']),
        adv_epsilon=float(g['adv_epsilon']), semaphore=sem, model_id=cfg['model']['base_model'])
        for t, s in zip(tasks, seeds)))
    return [rec for grp in groups for rec in grp]


def settings_from_config(cfg: dict) -> env_client.EnvSettings:
    e = cfg['env']
    return env_client.EnvSettings(prepared_roots=dict(e['prepared_roots']), run_root=e['run_root'],
                                  backend=e['backend'], image=e.get('image'), reward_version=e['reward_version'],
                                  max_turns=int(e['max_turns']), history_chars=int(e['history_chars']),
                                  tool_result_chars=int(e.get('tool_result_chars', 8000)))


def main(argv: Sequence[str] | None = None) -> int:
    from training.config import base_model_is_placeholder, load_config
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=str(Path(__file__).with_name('grpo_config.yaml')))
    ap.add_argument('--tasks', required=True, help='JSONL of TaskSpec dicts (one GRPO group each)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if base_model_is_placeholder(cfg):
        raise SystemExit('model.base_model is still the placeholder; set it (PHYS_GRPO__model__base_model=...)')
    tasks = [env_client.task_from_dict(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    rc, gen = cfg['rollout_client'], cfg['generation']
    policy = OpenAIChatPolicy(rc['base_url'], cfg['model']['base_model'], api_key=os.environ.get(rc['api_key_env']),
                              max_tokens=int(gen['max_tokens_per_turn']), temperature=float(gen['temperature']),
                              top_p=float(gen['top_p']), timeout_s=float(rc['request_timeout_s']))
    records = asyncio.run(run_batch(tasks=tasks, seeds=[args.seed] * len(tasks), cfg=cfg, policy=policy,
                                    settings=settings_from_config(cfg)))
    write_jsonl(records, args.out)
    print(json.dumps({'episodes': len(records), 'out': args.out}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
