-- Runs once, on first initialisation of an empty data volume.
-- pgvector ships compiled into the pgvector/pgvector image, so this is
-- just a matter of enabling it in our database.

CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram index support. Used later for keyword/lexical matching alongside
-- vector similarity when ranking feedback memories (hybrid retrieval).
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Deterministic UUIDs for trace ids and feedback ids.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

DO $$
BEGIN
  RAISE NOTICE 'pgvector version: %', (
    SELECT extversion FROM pg_extension WHERE extname = 'vector'
  );
END $$;
