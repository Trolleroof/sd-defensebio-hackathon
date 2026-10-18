"""Private, durable transcript. Never supplied as participant or judge context."""
import json
import sqlite3
import time
from pathlib import Path
from research_env.core.common import canonical, digest


class Transcript:
    def __init__(self, controller, db=None):
        root = Path(controller)
        root.mkdir(parents=True, exist_ok=True)
        self.owns_connection = db is None
        self.db = db if db is not None else sqlite3.connect(root/'controller.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS transcript (seq INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
        if self.owns_connection:
            self.db.commit()

    def append(self, kind, data):
        def insert():
            # Acquire the writer lock before reading the previous record. Provider
            # and controller can use separate connections to this same ledger.
            self.db.execute('UPDATE transcript SET payload=payload WHERE seq=(SELECT MAX(seq) FROM transcript)')
            row = self.db.execute('SELECT seq,payload FROM transcript ORDER BY seq DESC LIMIT 1').fetchone()
            seq = row[0]+1 if row else 0
            previous = json.loads(row[1])['sha256'] if row else None
            record = {'schema_version': 'physiology-transcript-v1', 'trace_id': f't{seq:06}',
                      'timestamp': time.time(), 'kind': kind, 'data': data, 'previous_sha256': previous}
            record['sha256'] = digest(record)
            self.db.execute('INSERT INTO transcript VALUES (?,?)', (seq, canonical(record).decode()))
            return record['trace_id']
        if self.owns_connection:
            with self.db:
                return insert()
        return insert()

    def records(self):
        previous = None
        records = []
        for i, (payload,) in enumerate(self.db.execute('SELECT payload FROM transcript ORDER BY seq')):
            record = json.loads(payload)
            body = {k:v for k,v in record.items() if k != 'sha256'}
            if record['trace_id'] != f't{i:06}' or record['previous_sha256'] != previous or digest(body) != record['sha256']:
                raise ValueError('Transcript hash chain corrupted')
            records.append(record)
            previous = record['sha256']
        return records

    def close(self):
        if self.owns_connection:
            self.db.close()
