from typing import Annotated, Any, Dict, List, Literal, TypedDict

from pydantic import BaseModel, Field

PLAN_MIN_ITEMS = 1
PLAN_MAX_ITEMS = 5
QueryMode = Literal["local", "global", "hybrid"]


class SubQuestionQueryPlan(BaseModel):
    sub_questions: Annotated[
        List[str],
        Field(
            min_length=PLAN_MIN_ITEMS,
            max_length=PLAN_MAX_ITEMS,
            description="把用户问题拆成1到5个更具体、互不重复、适合直接检索的子问题",
        ),
    ]
    query_modes: Annotated[
        List[QueryMode],
        Field(
            min_length=PLAN_MIN_ITEMS,
            max_length=PLAN_MAX_ITEMS,
            description="子问题对应的查询模式",
        ),
    ]
    query_topks: Annotated[
        List[int],
        Field(
            min_length=PLAN_MIN_ITEMS,
            max_length=PLAN_MAX_ITEMS,
            description="子问题对应的 top_k",
        ),
    ]
    query_chunk_topks: Annotated[
        List[int],
        Field(
            min_length=PLAN_MIN_ITEMS,
            max_length=PLAN_MAX_ITEMS,
            description="子问题对应的 chunk_top_k",
        ),
    ]


class AdaptiveQueryPlan(BaseModel):
    complexity: Literal["simple", "complex"] = Field(
        description="问题复杂度分类:simple 表示单问题检索即可,complex 表示需要多步拆分",
    )
    reason: str = Field(description="规划理由，简短即可")
    sub_questions: Annotated[
        List[str],
        Field(
            min_length=PLAN_MIN_ITEMS,
            max_length=PLAN_MAX_ITEMS,
            description="simple 时返回1个检索问题;complex 时返回1到5个子问题",
        ),
    ]
    query_modes: Annotated[
        List[QueryMode],
        Field(
            min_length=PLAN_MIN_ITEMS,
            max_length=PLAN_MAX_ITEMS,
            description="与 sub_questions 一一对应的查询模式",
        ),
    ]
    query_topks: Annotated[
        List[int],
        Field(
            min_length=PLAN_MIN_ITEMS,
            max_length=PLAN_MAX_ITEMS,
            description="与 sub_questions 一一对应的 top_k",
        ),
    ]
    query_chunk_topks: Annotated[
        List[int],
        Field(
            min_length=PLAN_MIN_ITEMS,
            max_length=PLAN_MAX_ITEMS,
            description="与 sub_questions 一一对应的 chunk_top_k",
        ),
    ]


class QuestionComplexity(BaseModel):
    complexity: Literal["simple", "complex"] = Field(
        description="问题复杂度分类:simple 表示可直接检索回答,complex 表示需要拆分多步检索",
    )
    reason: str = Field(description="分类理由，简短即可")


class SimpleQueryPlan(BaseModel):
    mode: QueryMode = Field(description="simple 问题的检索模式")
    top_k: int = Field(description="simple 问题的 top_k")
    chunk_top_k: int = Field(description="simple 问题的 chunk_top_k")


class EvidenceCheck(BaseModel):
    sufficient: bool = Field(description="当前检索结果是否足以回答该子问题")
    reason: str = Field(description="判断理由，简短即可")
    rewritten_question: str = Field(
        default="",
        description="若证据不足，给一个更容易检索的改写问题；若已充分则留空",
    )


class State(TypedDict, total=False):
    question: str
    requested_mode: str
    detected_complexity: str
    question_complexity: str
    effective_strategy: str
    planning_reason: str
    sub_questions: List[Any]
    query_modes: List[str]
    query_topks: List[int]
    query_chunk_topks: List[int]
    query_results: List[Dict[str, str]]
    subquery_tasks: List[Dict[str, Any]]
    subquery_results: List[Dict[str, str]]
    allowed_subject_ids: List[str]
    subject_working_dirs: Dict[str, str]
    response_language: str
    query_attempt: int
    needs_retry: bool
    insufficient_indices: List[int]
    insufficient_subquestion_ids: List[str]
    query_total_ms: str
    final_answer: str
