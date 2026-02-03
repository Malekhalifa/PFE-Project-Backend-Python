import os
from databases import Database
from dotenv import load_dotenv
import json

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
database = Database(DATABASE_URL)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_logs (
    timestamp TIMESTAMP NOT NULL DEFAULT now(),
    user_name TEXT NOT NULL,
    action TEXT NOT NULL,
    file_name TEXT NOT NULL,
    extension TEXT NOT NULL,
    file_size BIGINT NOT NULL
);
"""

async def connect():
    await database.connect()
    await database.execute(CREATE_TABLE_SQL)

async def disconnect():
    await database.disconnect()
