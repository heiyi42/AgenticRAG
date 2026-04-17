from __future__ import annotations

from agenticRAG.agentic_runtime import llm
from agenticRAG.instant_schema import InstantRoute

llm_instant_route_struct = llm.with_structured_output(InstantRoute)


def _build_route_prompt(question: str) -> str:
    return (
        "你是 GraphRAG 路由器。请在 local/global/hybrid 中选择一个最合适的查询模式。\n"
        "规则：\n"
        "- 具体事实、细节定位：local\n"
        "- 概括总结、关系脉络、时间线：global\n"
        "- 需要综合局部事实与整体脉络：hybrid\n"
        "只输出结构化字段。\n\n"
        f"问题：{question}"
    )


async def route_query_mode_async(question: str) -> tuple[str, str]:
    prompt = _build_route_prompt(question)
    obj: InstantRoute = await llm_instant_route_struct.ainvoke(prompt)
    return obj.mode, obj.reason
