"""verl agent loop that runs one physiology env episode per GRPO sample (GPU image only).

Coded against verl v0.9.1 (``verl/experimental/agent_loop/agent_loop.py``, ``tool_agent_loop.py``,
``tool_parser.py``). Registered through ``training/agent_loop.yaml`` via
``actor_rollout_ref.rollout.agent.agent_loop_config_path``.

How the pieces map onto verl:
* Group construction: each dataset row is one TaskSpec; verl repeats it ``rollout.n`` (= G) times
  and tags the copies with one uid, so the G episodes share the task, env settings and env seed.
* Generation: ``self.server_manager.generate`` hits verl's in-process async vLLM server, whose
  weights verl syncs from the FSDP actor every step (hybrid engine). No separate weight sync code.
* Tokens / masking: verl's Continuous Token builder (``ct_build_initial_tokens``,
  ``ct_merge_assistant_token``, ``ct_merge_context_msg``) produces ``response_mask`` = 1 for
  model-generated tokens and 0 for tool/observation tokens, so tool output never gets gradient.
* Reward: the env's terminal reward is returned as ``AgentLoopOutput.reward_score``; verl puts it
  on the last response token (``rm_scores``) and skips its own reward computation for that sample.
  ``algorithm.adv_estimator=grpo`` then computes the group-relative advantage per uid.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser

from training import env_client
from training.config import load_config
from training.rollout import AssistantTurn, PolicyExhausted, group_id_for, run_episode, settings_from_config
from training.trajectory_io import write_jsonl

from training.verl_overrides import AGENT_NAME


class VerlTokenPolicy:
    """Policy adapter that generates through verl's server manager and tracks token-level masks."""

    def __init__(self, loop: AgentLoopBase, sampling_params: dict, response_length: int, tool_format: str):
        self.loop, self.sampling_params, self.response_length = loop, sampling_params, response_length
        self.parser = ToolParser.get_tool_parser(tool_format, loop.tokenizer)
        self.request_id = uuid.uuid4().hex
        self.prompt_ids: Optional[list[int]] = None
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.n_seen = 0
        self.assistant_turns = 0

    async def step(self, messages: list[dict], tools: list[dict], *, sample_index: int) -> AssistantTurn:
        api_messages = _api(messages)
        if self.prompt_ids is None:
            self.prompt_ids = await self.loop.ct_build_initial_tokens(api_messages, tools=tools)
        else:
            previous = api_messages[: self.n_seen + 1]  # context through our last assistant message
            merge, mask, lps = await self.loop.ct_merge_context_msg(
                previous, api_messages, self.prompt_ids, self.response_mask,
                self.response_logprobs or None, tools=tools)
            self.prompt_ids, self.response_mask = merge.token_ids, mask
            if lps is not None:
                self.response_logprobs = lps
        if len(self.response_mask) >= self.response_length:
            raise PolicyExhausted('response token budget exhausted')
        self.n_seen = len(api_messages)
        params = dict(self.sampling_params)
        if self.parser.stop_token_ids:
            params['stop_token_ids'] = list(params.get('stop_token_ids') or []) + self.parser.stop_token_ids
        out = await self.loop.server_manager.generate(request_id=self.request_id, prompt_ids=self.prompt_ids,
                                                      sampling_params=params)
        merge, mask, lps = await self.loop.ct_merge_assistant_token(
            self.prompt_ids, out.token_ids, self.response_mask, self.response_logprobs or None,
            assistant_logprobs=out.log_probs or None)
        self.prompt_ids, self.response_mask = merge.token_ids, mask
        if lps is not None:
            self.response_logprobs = lps
        self.assistant_turns += 1
        content, calls = await self.parser.extract_tool_calls(out.token_ids, _tool_schemas(tools))
        return AssistantTurn(content or '', [
            {'id': c.tool_call_id or f'call_{self.assistant_turns}_{k}', 'type': 'function',
             'function': {'name': c.name, 'arguments': c.arguments}} for k, c in enumerate(calls)])

    def output(self, reward: float, num_turns: int, extra: dict) -> AgentLoopOutput:
        n = len(self.response_mask)
        response_ids = self.prompt_ids[-n:] if n else []
        prompt_ids = self.prompt_ids[: len(self.prompt_ids) - n]
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=response_ids[: self.response_length],
            response_mask=self.response_mask[: self.response_length],
            response_logprobs=self.response_logprobs[: self.response_length] if self.response_logprobs else None,
            reward_score=float(reward), num_turns=num_turns, extra_fields=extra)


def _api(messages: list[dict]) -> list[dict]:
    from training.rollout import to_api_messages
    return to_api_messages(messages)


def _tool_schemas(tools: list[dict]):
    try:  # parsers that type-cast arguments (qwen3_coder) want verl's schema objects
        from verl.tools.schemas import OpenAIFunctionToolSchema
        return [OpenAIFunctionToolSchema.model_validate(t) for t in tools]
    except Exception:
        return None


@register(AGENT_NAME)
class PhysiologyEnvAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phys_cfg = load_config(os.environ.get('PHYS_GRPO_CONFIG', Path(__file__).with_name('grpo_config.yaml')))
        self.settings = settings_from_config(self.phys_cfg)
        self.response_length = int(self.rollout_config.get('response_length', None)
                                   or self.config.data.max_response_length)
        self.tool_format = str(self.rollout_config.multi_turn.format)
        self.trajectory_dir = Path(self.phys_cfg['rollout_client']['output_dir'])

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get('extra_info') or {}
        task = env_client.task_from_dict(extra['task_spec'])
        env_client.refuse_final_evaluation(task)
        seed = int(extra.get('env_seed', 0))
        gid = group_id_for(task, seed, extra.get('group_tag', 0))
        sample_dir = Path(self.settings.run_root) / gid / f'sample-{uuid.uuid4().hex[:12]}'
        env = env_client.make_env(replace(self.settings, run_root=str(sample_dir)))
        policy = VerlTokenPolicy(self, sampling_params, self.response_length, self.tool_format)
        try:
            result = await run_episode(env=env, task=task, seed=seed, policy=policy, settings=self.settings,
                                       group_id=gid, sample_index=-1, model_id=self.phys_cfg['model']['base_model'])
        finally:
            env_client.close(env)
        write_jsonl([result.record], self.trajectory_dir / f'verl-{os.getpid()}.jsonl')
        return policy.output(result.reward, num_turns=len(result.record['messages']), extra={
            'physiology_group_id': gid, 'physiology_invalid_actions': result.invalid_actions,
            'physiology_forced_termination': result.forced_termination,
            'physiology_reward_components': json.dumps(result.record['reward_components'], default=str)})
