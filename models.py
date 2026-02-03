# Run this once to create table
CREATE_EXTENSION_IF_NOT_EXISTS = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
"""

CREATE_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP NOT NULL DEFAULT now(),
    level VARCHAR(10),
    action VARCHAR(100),
    entity VARCHAR(50),
    entity_id VARCHAR(100),
    message TEXT,
);
"""
