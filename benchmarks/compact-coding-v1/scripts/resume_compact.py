"""Audited accounting amendment; retain frozen inference requests and sealed results."""
from pathlib import Path
import json
from research_env import compact
from research_env.core.common import read,write,file_hash
from research_env.core.spend import SpendLedger

class ProviderAwareLedger(SpendLedger):
    """Reserve xAI's full advertised ceiling after an observed max_tokens violation."""
    def __init__(self,path,limit_usd):
        super().__init__(path,limit_usd)
        self.output_limits={}
    def reserve(self,body,pricing):
        reserved=dict(body)
        if body['model']=='x-ai/grok-4.7':
            reserved['max_tokens']=max(body['max_tokens'],450000)
        identifier=super().reserve(reserved,pricing)
        self.output_limits[identifier]=body['max_tokens']
        return identifier
    def settle(self,identifier,usage):
        super().settle(identifier,usage)
        reported=usage.get('completion_tokens')
        requested=self.output_limits.get(identifier)
        if requested is not None and type(reported) is int and reported>requested:
            raise ValueError('Provider exceeded requested output budget; return one concise tool call')


def resume(key):
    root=Path('.leaderboard-runs/compact-eight')
    # The same persistent ledger is reused; no previous charge or receipt is reset.
    amendment={'version':'provider-accounting-amendment-v1','reason':'User requested continuation after xAI exceeded its requested token cap',
      'implementation_sha256':file_hash(Path(__file__)),
      'requested_output_tokens':1536,'xai_reserved_max_output_tokens':450000,
      'total_budget_usd':16,'inference_request_changes':False,
      'scope':'Conservative cost reservation and rejection of over-budget responses; unchanged frozen model requests, cohorts and scoring',
      'bound_notice':'The previous $15.12 whole-run reservation estimate depended on provider compliance and is no longer a completion guarantee. The shared $16 ledger remains enforced.'}
    path=root/'accounting-amendment.json'
    if path.exists() and read(path)!=amendment:raise ValueError('Accounting amendment changed')
    write(path,amendment)
    old=root/'runs/x-ai--grok-4-7-medium/heat/seed-0/api/experiment0-0.json'
    prior=read(old)
    if prior['status']=='running' and prior.get('usage',{}).get('completion_tokens',0)>1536:
        write(root/'accounting-overrun-original.json',prior)
        prior.update(status='failed',error='ProviderOutputBudgetExceeded',
          correction='Provider exceeded requested output budget. Return one concise respond tool call, reason <=160 characters, short executable code.')
        write(old,prior)
    compact.SpendLedger=ProviderAwareLedger
    return compact.execute('leaderboard/sweeps/compact-eight/plan.json',root,key)
