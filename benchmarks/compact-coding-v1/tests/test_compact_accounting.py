import tempfile,unittest
from pathlib import Path
from scripts.resume_compact import ProviderAwareLedger
from research_env.core.spend import SpendLimitReached

class Accounting(unittest.TestCase):
 def test_same_cap_full_provider_reservation_and_actual_charge(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'spend.sqlite';ledger=ProviderAwareLedger(path,16)
   body={'model':'x-ai/grok-4.7','max_tokens':1536,'messages':[]}
   charge=ledger.reserve(body,{'prompt':.000002,'completion':.000006})
   self.assertGreater(ledger.summary()['accounted_usd'],2.7)
   self.assertEqual(body['max_tokens'],1536)
   with self.assertRaisesRegex(ValueError,'output budget'):
    ledger.settle(charge,{'cost':.438316,'completion_tokens':72797})
   self.assertAlmostEqual(ledger.summary()['accounted_usd'],.438316)
   ledger.close();ledger=ProviderAwareLedger(path,16)
   self.assertAlmostEqual(ledger.summary()['accounted_usd'],.438316)
   with self.assertRaises(ValueError):ProviderAwareLedger(path,17)
   ledger.close()
 def test_guard_blocks_when_advertised_ceiling_wont_fit(self):
  with tempfile.TemporaryDirectory() as d:
   ledger=ProviderAwareLedger(Path(d)/'spend.sqlite',1)
   with self.assertRaises(SpendLimitReached):ledger.reserve({'model':'x-ai/grok-4.7','max_tokens':1536},{'prompt':.000002,'completion':.000006})
   self.assertEqual(ledger.summary()['requests'],0);ledger.close()

if __name__=='__main__':unittest.main()
