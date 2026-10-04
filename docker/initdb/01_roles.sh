#!/bin/bash
# ============================================================
# Read-only role for agent-generated SQL.
#
# WHY THIS EXISTS
# The agent writes its own SQL from an LLM prompt. We validate that SQL before
# running it (single statement, SELECT only, parsed with sqlglot), but a
# validator is just a parser and parsers can be fooled. This role is the
# second, independent line of defence: even if a DELETE slipped past the
# validator, Postgres itself would refuse to execute it.
#
# Two separate connections exist in this project:
#   DATABASE_URL     -> analyst  (read-write: seeding, feedback memory, traces)
#   DATABASE_URL_RO  -> agent_ro (read-only: every query the agent generates)
#
# This is a .sh rather than a .sql file so the password comes from the
# environment. A .sql file would mean committing a credential to git.
# ============================================================
set -euo pipefail

if [ -z "${AGENT_RO_PASSWORD:-}" ]; then
  echo "FATAL: AGENT_RO_PASSWORD is not set. Check .env and docker-compose.yml." >&2
  exit 1
fi

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v ro_password="$AGENT_RO_PASSWORD" \
     -v db_name="$POSTGRES_DB" \
     -v rw_user="$POSTGRES_USER" <<-'EOSQL'

    CREATE ROLE agent_ro WITH LOGIN PASSWORD :'ro_password';

    -- Belt and braces: every transaction this role opens is read-only by
    -- default, blocking writes at the engine level regardless of privileges.
    ALTER ROLE agent_ro SET default_transaction_read_only = on;

    -- A runaway analytical query can't hang the agent forever.
    ALTER ROLE agent_ro SET statement_timeout = '15s';

    -- One bad query shouldn't eat all the memory.
    ALTER ROLE agent_ro SET work_mem = '16MB';

    GRANT CONNECT ON DATABASE :"db_name" TO agent_ro;
    GRANT USAGE ON SCHEMA public TO agent_ro;

    -- SELECT on everything that exists now...
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_ro;

    -- ...and on everything the seeding scripts create later. Without this,
    -- tables created after this script runs are invisible to agent_ro.
    ALTER DEFAULT PRIVILEGES IN SCHEMA public
      GRANT SELECT ON TABLES TO agent_ro;

    -- Remove the implicit CREATE grant that PUBLIC has on schema public,
    -- so agent_ro cannot create scratch objects.
    REVOKE CREATE ON SCHEMA public FROM PUBLIC;
    GRANT CREATE ON SCHEMA public TO :"rw_user";
EOSQL

echo "[initdb] agent_ro role created (read-only, 15s statement timeout)"
