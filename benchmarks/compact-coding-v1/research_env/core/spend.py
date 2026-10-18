"""Persistent shared API spend guard; unknown/failed requests retain their ceiling."""
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import sqlite3
import uuid
from research_env.core.common import canonical, finite


class SpendLimitReached(RuntimeError):
    pass


class SpendLedger:
    def __init__(self,path,limit_usd):
        if not finite(limit_usd) or limit_usd<=0:
            raise ValueError('Explicit positive sweep spend cap required')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.limit=int(Decimal(str(limit_usd))*1_000_000_000)
        self.db.execute('CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, limit_nano INTEGER NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS charges (id TEXT PRIMARY KEY, model TEXT, ceiling_nano INTEGER, charged_nano INTEGER, accounting TEXT)')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO budget VALUES (1,?)',(self.limit,))
        if self.db.execute('SELECT limit_nano FROM budget WHERE id=1').fetchone()[0]!=self.limit:
            self.db.close()
            raise ValueError('Cannot silently change the persisted spend cap')

    def reserve(self,body,pricing):
        # UTF-8 bytes plus message overhead bound the input token count for these
        # text-only requests. Reserve the full output ceiling, including reasoning.
        prompt=Decimal(str(pricing['prompt']))
        completion=Decimal(str(pricing['completion']))
        if prompt<0 or completion<0 or not prompt.is_finite() or not completion.is_finite():
            raise ValueError('Invalid frozen token pricing')
        cost=(len(canonical(body))+4096)*prompt+body['max_tokens']*completion
        amount=int((cost*1_000_000_000).to_integral_value(rounding=ROUND_CEILING))
        identifier=uuid.uuid4().hex
        with self.db:
            self.db.execute('UPDATE budget SET limit_nano=limit_nano WHERE id=1')
            used=self.db.execute('SELECT COALESCE(SUM(charged_nano),0) FROM charges').fetchone()[0]
            if used+amount>self.limit:
                raise SpendLimitReached('Sweep API spend cap would be exceeded by the next request')
            self.db.execute('INSERT INTO charges VALUES (?,?,?,?,?)',
                            (identifier,body['model'],amount,amount,'conservative_reservation'))
        return identifier

    def settle(self,identifier,usage):
        cost=usage.get('cost')
        if not finite(cost) or cost<0:
            return  # Keep the reserved ceiling when actual cost is unavailable.
        amount=int((Decimal(str(cost))*1_000_000_000).to_integral_value(rounding=ROUND_CEILING))
        with self.db:
            row=self.db.execute('SELECT ceiling_nano FROM charges WHERE id=?',(identifier,)).fetchone()
            if row is None:
                raise ValueError('Unknown spend reservation')
            self.db.execute('UPDATE charges SET charged_nano=?, accounting=? WHERE id=?',
                            (amount,'provider_reported',identifier))
        if amount>row[0]:
            raise SpendLimitReached('Reported provider cost exceeded the frozen price ceiling; stop and audit')

    def summary(self):
        total=self.db.execute('SELECT COALESCE(SUM(charged_nano),0),COUNT(*) FROM charges').fetchone()
        unknown=self.db.execute("SELECT COUNT(*) FROM charges WHERE accounting!='provider_reported'").fetchone()[0]
        return {'limit_usd':self.limit/1e9,'accounted_usd':total[0]/1e9,'requests':total[1],
                'requests_with_reserved_cost':unknown,'remaining_usd':max(0,(self.limit-total[0])/1e9)}

    def close(self):
        self.db.close()
