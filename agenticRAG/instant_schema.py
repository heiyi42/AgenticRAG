from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class InstantRoute(BaseModel):
    mode: Literal["local", "global", "hybrid"] = Field(
        description="查询模式，只能是 local/global/hybrid"
    )
    reason: str = Field(description="简短说明为什么选择这个模式")


class InstantState(TypedDict):
    question: str
    route_mode: str
    route_reason: str
    answer: str
    elapsed_ms: str
    query_status: str
    query_message: str

