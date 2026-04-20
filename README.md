# AgenticRAG

AgenticRAG 是一个面向三门固定课程的 Web 学习助手，当前覆盖：

- `C_program`：C 语言
- `operating_systems`：操作系统
- `cybersec_lab`：网络安全实验

当前主路径是 `Flask + webapp_core 编排层 + LightRAG/OpenAI + Pydantic 结构化输出`。仓库里虽然保留了部分 `langgraph` 依赖和历史痕迹，但当前 Web 主链并不是一个以 LangGraph 为核心的运行时。

## 主要能力

- `Auto / Instant / DeepSearch` 三种回答模式
- 基于学科路由和检索网关的多课程问答
- 题目辅导与过程化解题
- C 代码分析、编译诊断与受限执行
- Web 会话管理、SSE 流式输出、摘要记忆

## 目录说明

- `webapp.py`：Flask 入口，负责创建应用、注册路由和启动 Web 服务
- `webapp_core/`：Web 编排层，包含路由判断、流式输出、会话管理、题目辅导、代码分析等模块
- `webapp_core/auto_runtime.py`：Auto 模式复用的路由、预算、评审与流式 instant 辅助
- `webapp_core/config.py`：Web 三种模式的超时、阈值和功能开关
- `agenticRAG/`：检索运行时、回答辅助函数、短时记忆与结构化 schema
- `templates/`：前端模板
- `data/`：原始课程资料、题库数据和聊天持久化文件
- `storage/`：每门课程对应的 LightRAG working dir，运行时直接读取
- `script/`：本地一键启动脚本
- `scripts/`：辅助脚本，例如题库 embedding 索引构建
- `tests/`：当前行为边界最清晰的回归测试
- `docs/`：产品说明与模块报告

## 运行前提

- Python `3.11`
- 可用的 OpenAI 兼容接口
- 已准备好的课程知识库 working dir，默认在 `./storage`

可选能力：

- C 编译器：`clang`、`gcc`、`cc` 或 `cl`
- Neo4j：仅用于图谱导入/可视化脚本，不是当前 Web 主运行时依赖

## 快速启动

1. 创建并激活 Python 3.11 环境。

```bash
conda create -n py311 python=3.11 -y
conda activate py311
```

2. 安装项目依赖。

```bash
pip install -e .
```

如果还需要 Neo4j / 图可视化等工具脚本：

```bash
pip install -e ".[tools]"
```

3. 准备环境变量。

```bash
cp .env.example .env
```

至少需要补上：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（如果你走兼容网关）

常用变量：

- `WEB_HOST` / `WEB_PORT`：Web 服务地址，默认 `127.0.0.1:7860`
- `WEB_STORAGE_ROOT`：课程知识库根目录，默认 `./storage`
- `WEB_CHAT_STORE_PATH`：聊天记录持久化文件，默认 `./data/web_chats.json`
- `WEB_CODE_ANALYSIS_COMPILER`：指定代码分析使用的编译器
- `QUESTION_BANK_EMBED_INDEX_PATH`：题库 embedding 索引路径

4. 确认运行时数据目录可用。

默认要求存在：

```text
storage/
  C_program/
  operating_systems/
  cybersec_lab/
```

如果这些目录缺失，检索链无法正常工作。

5. 启动 Web 应用。

```bash
python webapp.py
```

或使用仓库内启动脚本：

```bash
bash script/build_and_run.sh
```

启动后访问 [http://127.0.0.1:7860](http://127.0.0.1:7860)。

## 数据与索引

这个仓库里要分清三类东西：

1. 源码
   `webapp.py`、`webapp_core/`、`agenticRAG/`、`tests/`

2. 原始课程资料 / 题库
   例如 `data/subject_chapters/`、`data/tutoring_question_bank/`

3. 生成后的检索索引
   `storage/<subject>/` 下的 `vdb_*.json`、`kv_store_*.json`、`graph_chunk_entity_relation.graphml` 等文件

`storage/` 中的内容是运行时必需的，但它们更像“可重建的检索工作目录”，而不是适合放进普通 Git 历史的大型源码文件。团队协作时更推荐：

- 由脚本重建
- 单独打包分发
- 通过 Docker volume / 共享目录挂载

不建议直接把大体积 `storage/` 文件长期塞进普通 Git 历史。

## 题库 embedding 索引

题目辅导模块默认读取：

- `data/tutoring_question_bank/questions.jsonl`
- `data/tutoring_question_bank/questions.embedding_index.json`

如果需要重建题库 embedding 索引：

```bash
python scripts/build_question_bank_embeddings.py
```

## 测试

当前测试以 `unittest` 为主，推荐直接跑：

```bash
python -m unittest discover -s tests
```

如果你使用的是本地 `py311` Conda 环境，也可以：

```bash
conda run -n py311 python -m unittest discover -s tests
```

## 当前实现边界

- 这是三门固定课程的专用学习助手，不是开放域聊天机器人
- Web 运行时当前依赖本地 `LightRAG working_dir`
- Neo4j 相关代码目前主要用于图谱导入/展示，不替代当前 `storage/` 运行时
- 代码分析使用的是后端可用编译器环境，不是浏览器本地环境

## 参考文档

- [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)
- [产品说明书](./docs/AgenticRAG_Chat_产品说明书_当前版.html)
- [题目辅导模块报告](./docs/problem_tutoring_module_report.md)
