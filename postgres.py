from __future__ import annotations

from psycopg2 import connect
from dataclasses import dataclass


PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_HOST = "db"
PG_PORT = 5432

@dataclass
class PGInstance:
    pg_connect = connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        options="-c statement_timeout=300000",
    )

    def __post_init__(self):
        self.pg_connect.autocommit = True
        self.cursor = self.pg_connect.cursor()

    def close_connection(self):
        self.pg_connect.close()
