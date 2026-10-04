# Self-Improving Data Analyst Agent
#
#   make setup       one-time: venv + dependencies
#   make db-up       start Postgres + pgAdmin (starts Colima if needed)
#   make verify      check the whole stack end to end
#   make help        everything else

PY       := .venv/bin/python
PIP      := .venv/bin/pip
SCRATCH  := .venv/bin
SHELL    := /bin/bash

.DEFAULT_GOAL := help
.PHONY: help setup db-up db-down db-reset db-logs psql psql-ro pgadmin \
        verify verify-llm verify-db seed test run ui benchmark clean colima-up \
        llm-up llm-down llm-models llm-status up down \
        init-db init-memory generate load verify-effects snapshots

help:  ## Show this help
	@echo "Self-Improving Data Analyst Agent"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup

setup:  ## Create .venv and install dependencies
	@test -d .venv || /opt/homebrew/bin/python3.13 -m venv .venv
	@$(PIP) install --quiet --upgrade pip
	@$(PIP) install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env - add your GEMINI_API_KEY")
	@echo "setup done. Next: make db-up"

# ---------------------------------------------------------------- database

# Colima does not survive a reboot, so every db target depends on this.
# It is a no-op when the VM is already running.
colima-up:
	@colima status >/dev/null 2>&1 || { \
	  echo "Colima VM not running - starting it (takes ~30s)..."; \
	  colima start --cpu 2 --memory 4 --disk 20 --vm-type vz --mount-type virtiofs; \
	}

db-up: colima-up  ## Start Postgres + pgAdmin
	@docker compose up -d
	@echo ""
	@echo "  Postgres : localhost:5433"
	@echo "  pgAdmin  : http://localhost:5050"

db-down:  ## Stop containers (data is preserved)
	@docker compose down

db-reset:  ## DESTROY all data and re-init from scratch
	@echo "This deletes the database volume. All data, feedback memory and traces."
	@read -p "Type 'yes' to confirm: " c && [ "$$c" = "yes" ] || (echo "aborted"; exit 1)
	@docker compose down -v
	@docker compose up -d
	@echo "database reset. Run: make seed"

db-logs:  ## Tail Postgres logs
	@docker compose logs -f postgres

psql:  ## Open psql as the read-write user
	@docker exec -it sida_postgres psql -U analyst -d ecommerce

psql-ro:  ## Open psql as the read-only agent role
	@set -a; source .env; set +a; \
	docker exec -it -e PGPASSWORD="$$AGENT_RO_PASSWORD" sida_postgres \
	  psql -U agent_ro -d ecommerce

pgadmin:  ## Open pgAdmin in the browser
	@open http://localhost:5050

# ---------------------------------------------------------------- local LLM

# Models the project needs locally. Kept here so `make llm-models` is the
# single source of truth for what to pull on a fresh machine.
OLLAMA_CHAT_MODEL  := granite4.2:8b
OLLAMA_EMBED_MODEL := qwen3-embedding:0.6b

llm-up:  ## Start the Ollama server if it isn't already running
	@curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1 && \
	  echo "Ollama already running" || { \
	    echo "Starting Ollama..."; \
	    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
	      nohup ollama serve > /tmp/ollama.log 2>&1 & \
	    for i in $$(seq 1 30); do \
	      curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break; \
	      sleep 1; \
	    done; \
	    curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1 \
	      && echo "Ollama up" \
	      || { echo "Ollama failed to start - see /tmp/ollama.log"; exit 1; }; \
	  }

llm-down:  ## Stop the Ollama server
	@pkill -f "ollama serve" 2>/dev/null && echo "Ollama stopped" || echo "Ollama was not running"

llm-models: llm-up  ## Pull the local models this project needs
	@ollama pull $(OLLAMA_CHAT_MODEL)
	@ollama pull $(OLLAMA_EMBED_MODEL)
	@ollama list

llm-status:  ## Show installed models and what is loaded in memory
	@ollama list
	@echo ""
	@ollama ps

# ---------------------------------------------------------------- verify

verify: verify-db verify-llm  ## Check the entire stack

verify-db: colima-up  ## Check Postgres, pgvector and the read-only role
	@$(PY) scripts/verify_db.py

verify-llm: llm-up  ## Check the LLM provider, models and embedding dimension
	@$(PY) scripts/verify_llm.py

# ---------------------------------------------------------------- combined

up: db-up llm-up  ## Start everything (database + local LLM)

down: db-down llm-down  ## Stop everything

# ---------------------------------------------------------------- app
# (targets below are implemented in later phases)

init-db:  ## Create the business schema (DESTROYS public tables, keeps memory)
	@$(PY) scripts/init_db.py

init-memory:  ## Create the memory schema (refuses if feedback exists; FORCE=1 overrides)
	@$(PY) scripts/init_memory.py

generate:  ## Generate the synthetic CSVs from the effects manifest
	@$(PY) scripts/generate_data.py

load:  ## Load the generated CSVs into Postgres
	@$(PY) scripts/seed_database.py

verify-effects:  ## Measure the planted effects and write ground truth
	@$(PY) scripts/verify_effects.py

seed: init-db generate load verify-effects  ## Full data pipeline, end to end
	@echo ""
	@echo "data pipeline complete"

test:  ## Run the test suite
	@$(PY) -m pytest -q

run:  ## Start the FastAPI server
	@$(PY) -m uvicorn app.api.main:app --reload --port 8000

ui:  ## Start the Streamlit Learning Lab
	@$(SCRATCH)/streamlit run ui/streamlit_app.py

snapshots: llm-up  ## Build the clean/poisoned/mixed memory snapshots
	@$(PY) scripts/build_snapshots.py

benchmark: llm-up  ## Run the benchmark (ARGS="--quick" for a smoke test)
	@$(PY) scripts/run_benchmark.py $(ARGS)

clean:  ## Remove caches and build artefacts
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	@echo "cleaned"
