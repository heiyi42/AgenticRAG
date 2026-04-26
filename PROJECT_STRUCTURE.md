# Project Structure

## Core App
- `webapp.py`: Flask web entrypoint
- `webapp_core/`: web app service/config/session modules
- `webapp_core/graph_service.py`: Neo4j graph query adapter for visualization APIs
- `frontend/`: React + TypeScript visualization workbench
- `templates/`: legacy Flask template files served at `/legacy`

## RAG Core
- `agenticRAG/`: retrieval/runtime helpers and chain nodes
- `data/`: source text and chat persistence (`web_chats.json`)
- `storage/`: subject-specific LightRAG working dirs and generated graph/index data

## Mode Modules
- `webapp_core/auto_runtime.py`: Web Auto 模式复用的路由/评审辅助函数与默认参数
- `webapp_core/config.py`: Web 三种模式的超时与运行配置

## Tools
- `utils/graph_visual_with_neo4j.py`: imports LightRAG GraphML/docs/chunks into Neo4j
- `scripts/build_question_bank_embeddings.py`: rebuilds tutoring question-bank embeddings

## Tests / Docs
- `tests/`: unittest regression suite
- `docs/`: product manual and tutoring module report

## Notes
- Removed redundant cache files: `__pycache__/`, `*.pyc`, `.DS_Store`.
- Main runtime files were kept in place to avoid breaking imports.
