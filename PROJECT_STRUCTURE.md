# Project Structure

## Core App
- `webapp.py`: Flask web entrypoint
- `webapp_core/`: web app service/config/session modules
- `templates/`: frontend template files

## RAG Core
- `agenticRAG/`: retrieval/runtime helpers and chain nodes
- `dickens/`: LightRAG working dir / index data
- `data/`: source text and chat persistence (`web_chats.json`)

## CLI Tools
- `models/auto.py`: Auto mode CLI
- `models/instant.py`: Instant mode CLI
- `models/deep_search.py`: DeepSearch mode CLI
- `benchmark_query_modes.py`: mode latency benchmark
- `run_eval_compare.py`: batch eval runner

## Assets / Notebook / Legacy
- `artifacts/`: generated visual assets (e.g. graph html)
- `notebooks/`: notebook experiments
- `legacy/`: archived old scripts/demo code not used by web main flow

## Notes
- Removed redundant cache files: `__pycache__/`, `*.pyc`, `.DS_Store`.
- Main runtime files were kept in place to avoid breaking imports.
