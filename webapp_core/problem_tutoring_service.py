from __future__ import annotations

import asyncio
import ast
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


SubjectId = Literal["C_program", "operating_systems", "cybersec_lab", "unknown"]


class TutoringProblemAnalysis(BaseModel):
    is_problem: bool = Field(description="是否属于课程题目辅导/过程化解题请求")
    subject_id: SubjectId = Field(description="题目所属学科ID")
    problem_type: str = Field(description="题型ID，例如 c_output / os_page_replacement")
    confidence: float = Field(ge=0.0, le=1.0, description="题型识别置信度")
    target: str = Field(default="", description="题目要求求解的目标")
    extracted_conditions: list[str] = Field(
        default_factory=list,
        description="从题干中抽取出的关键条件",
    )
    knowledge_points: list[str] = Field(
        default_factory=list,
        description="涉及的知识点",
    )
    answer_focus: Literal[
        "calculation",
        "code_reading",
        "debugging",
        "conceptual",
        "lab_analysis",
        "general",
    ] = Field(default="general", description="答案组织重点")
    reason: str = Field(default="", description="识别理由")


@dataclass(frozen=True)
class ProblemTemplate:
    id: str
    subject_id: str
    name: str
    problem_type: str
    aliases: tuple[str, ...]
    knowledge_keywords: tuple[str, ...]
    steps: tuple[str, ...]
    output_sections: tuple[str, ...]
    solver_hint: str
    priority: int = 50

    def to_prompt_block(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProblemTutoringService:
    DEFAULT_QUESTION_BANK_PATH = Path("data/tutoring_question_bank/questions.jsonl")
    DEFAULT_QUESTION_BANK_EMBEDDING_INDEX_PATH = Path(
        "data/tutoring_question_bank/questions.embedding_index.json"
    )
    SUBJECT_LABELS = {
        "C_program": "C语言",
        "operating_systems": "操作系统",
        "cybersec_lab": "网络安全实验",
        "unknown": "未知学科",
    }

    TEMPLATES: tuple[ProblemTemplate, ...] = (
        ProblemTemplate(
            id="c_output",
            subject_id="C_program",
            name="C语言程序输出题",
            problem_type="c_output",
            aliases=("输出什么", "运行结果", "程序输出", "printf", "代码输出"),
            knowledge_keywords=("printf", "变量跟踪", "表达式求值", "控制流", "数组", "指针"),
            steps=(
                "确认代码能否通过语法检查；若题干只问输出，不假装实际运行。",
                "按执行顺序跟踪变量、数组/指针指向、循环次数和分支条件。",
                "把每次关键状态变化写成表格或分点过程。",
                "最后给出标准输出，并解释换行、空格、未定义行为等边界。",
            ),
            output_sections=("题型判断", "考点定位", "手动跟踪过程", "最终输出", "易错点", "类题训练方向"),
            solver_hint="以手动执行轨迹为主；若存在 UB、越界或未初始化变量，先指出不能给稳定输出。",
            priority=95,
        ),
        ProblemTemplate(
            id="c_debug",
            subject_id="C_program",
            name="C语言找错改错题",
            problem_type="c_debug",
            aliases=("找错", "改错", "报错", "哪里错", "debug", "段错误"),
            knowledge_keywords=("编译错误", "运行错误", "指针", "数组越界", "生命周期", "类型转换"),
            steps=(
                "先区分编译错误、运行错误、逻辑错误或未定义行为。",
                "定位最可能的错误语句和触发条件。",
                "给出最小修改方案，并说明为什么这样改。",
                "补充相同错误的识别方法和防御式写法。",
            ),
            output_sections=("题型判断", "错误定位", "修改方案", "原因说明", "排查建议", "类题训练方向"),
            solver_hint="不要凭空编造编译器诊断；没有工具结果时，用代码语义分析说明“最可能”。",
            priority=90,
        ),
        ProblemTemplate(
            id="c_pointer_array",
            subject_id="C_program",
            name="C语言指针/数组分析题",
            problem_type="c_pointer_array",
            aliases=("指针", "数组", "解引用", "地址", "下标", "数组名退化"),
            knowledge_keywords=("指针算术", "数组退化", "解引用", "地址", "越界", "字符串"),
            steps=(
                "先画出数组元素、地址关系和指针当前指向的位置。",
                "逐步分析 p+i、*(p+i)、a[i]、&a[i] 等表达式含义。",
                "检查是否越界、是否解引用空指针/野指针/悬空指针。",
                "再给出题目要求的值、输出或错误判断。",
            ),
            output_sections=("题型判断", "考点定位", "地址/指向关系", "逐步分析", "结论", "易错点"),
            solver_hint="把“地址移动”和“取值”分开解释，避免把数组名、数组首元素地址、整个数组地址混为一谈。",
            priority=85,
        ),
        ProblemTemplate(
            id="c_function_call",
            subject_id="C_program",
            name="C语言函数调用与参数传递题",
            problem_type="c_function_call",
            aliases=("函数调用", "参数传递", "形参", "实参", "返回值", "递归"),
            knowledge_keywords=("值传递", "指针参数", "作用域", "生命周期", "递归栈"),
            steps=(
                "区分形参、实参、局部变量和返回值。",
                "按调用顺序记录每一层函数调用的参数值。",
                "若有指针参数，说明修改的是指针本身还是指向对象。",
                "回到主调函数后再判断最终输出或变量值。",
            ),
            output_sections=("题型判断", "调用关系", "参数变化过程", "结论", "易错点"),
            solver_hint="函数参数默认是值传递；只有通过地址/指针间接访问时，才会影响调用者对象。",
            priority=75,
        ),
        ProblemTemplate(
            id="os_page_replacement",
            subject_id="operating_systems",
            name="页面置换题",
            problem_type="os_page_replacement",
            aliases=("页面置换", "缺页", "页框", "FIFO", "LRU", "OPT", "Clock", "命中率"),
            knowledge_keywords=("虚拟内存", "页面置换算法", "缺页中断", "页框", "FIFO", "LRU", "OPT"),
            steps=(
                "抽取页面访问序列、页框数、初始状态和指定算法。",
                "按访问序列逐步判断命中或缺页。",
                "缺页时按照算法规则选择被置换页面，并记录页框状态。",
                "统计缺页次数、命中次数、缺页率/命中率，并给出过程表。",
            ),
            output_sections=("题型判断", "条件抽取", "算法规则", "过程表", "计算结果", "考试写法", "易错点"),
            solver_hint="优先用表格展示每次访问后的页框状态；没有页框数或访问序列时必须说明条件不足。",
            priority=95,
        ),
        ProblemTemplate(
            id="os_cpu_scheduling",
            subject_id="operating_systems",
            name="进程调度计算题",
            problem_type="os_cpu_scheduling",
            aliases=("进程调度", "FCFS", "SJF", "SRTF", "RR", "时间片", "周转时间", "等待时间"),
            knowledge_keywords=("短程调度", "FCFS", "SJF", "SRTF", "RR", "响应比", "周转时间", "等待时间"),
            steps=(
                "抽取进程到达时间、服务时间、优先级、时间片和调度算法。",
                "按时间轴模拟调度顺序，必要时画甘特图。",
                "计算完成时间、周转时间、带权周转时间、等待时间。",
                "汇总平均值，并说明抢占/非抢占规则。",
            ),
            output_sections=("题型判断", "条件抽取", "调度过程", "计算表", "结果", "考试写法", "易错点"),
            solver_hint="时间片轮转和抢占式算法必须逐时间段模拟；不要跳过到达时间导致的空闲区间。",
            priority=92,
        ),
        ProblemTemplate(
            id="os_banker",
            subject_id="operating_systems",
            name="银行家算法题",
            problem_type="os_banker",
            aliases=("银行家算法", "安全序列", "Need", "Available", "Max", "Allocation"),
            knowledge_keywords=("死锁避免", "安全状态", "Need矩阵", "Available向量", "安全序列"),
            steps=(
                "抽取 Available、Max、Allocation 和 Request。",
                "计算 Need = Max - Allocation。",
                "执行安全性算法，逐轮寻找 Need <= Work 的进程。",
                "判断是否存在安全序列；若有请求，再判断试分配后是否安全。",
            ),
            output_sections=("题型判断", "条件抽取", "Need计算", "安全性检查", "结论", "考试写法", "易错点"),
            solver_hint="矩阵条件不完整时不能硬算；每轮 Work 和 Finish 的变化要写清楚。",
            priority=90,
        ),
        ProblemTemplate(
            id="os_deadlock",
            subject_id="operating_systems",
            name="死锁分析题",
            problem_type="os_deadlock",
            aliases=("死锁", "资源分配图", "安全状态", "环路", "饥饿"),
            knowledge_keywords=("互斥", "占有并等待", "不可抢占", "循环等待", "资源分配图"),
            steps=(
                "识别进程、资源、已分配资源和请求资源。",
                "检查死锁四个必要条件是否同时成立。",
                "若有资源分配图，分析是否存在环以及每类资源实例数。",
                "给出死锁判断、解除或预防方案。",
            ),
            output_sections=("题型判断", "条件抽取", "四条件分析", "结论", "处理策略", "易错点"),
            solver_hint="单实例资源图有环通常可判死锁；多实例资源有环不一定死锁，需要进一步检查。",
            priority=86,
        ),
        ProblemTemplate(
            id="os_pv_sync",
            subject_id="operating_systems",
            name="PV同步互斥题",
            problem_type="os_pv_sync",
            aliases=("PV", "P操作", "V操作", "信号量", "同步", "互斥", "生产者消费者"),
            knowledge_keywords=("临界区", "互斥", "同步", "信号量", "P操作", "V操作", "前驱关系"),
            steps=(
                "区分互斥关系和同步先后关系。",
                "为共享资源设置互斥信号量，为前驱约束设置同步信号量。",
                "确定每个信号量初值。",
                "把 P/V 操作放到进入临界区前、离开临界区后或前驱/后继边界处。",
            ),
            output_sections=("题型判断", "关系分析", "信号量设计", "伪代码", "正确性说明", "易错点"),
            solver_hint="互斥信号量初值通常为 1；同步信号量初值通常按可先发生的事件数量设置。",
            priority=85,
        ),
        ProblemTemplate(
            id="cybersec_lab_steps",
            subject_id="cybersec_lab",
            name="网络安全实验步骤说明题",
            problem_type="cybersec_lab_steps",
            aliases=("实验步骤", "怎么做", "配置", "搭建", "验证"),
            knowledge_keywords=("实验目标", "实验环境", "操作步骤", "结果验证", "安全边界"),
            steps=(
                "明确实验目标、授权环境和前置条件。",
                "按准备、配置、执行、观察、验证的顺序组织步骤。",
                "说明每一步预期现象和判断标准。",
                "补充常见失败原因和排查方式。",
            ),
            output_sections=("题型判断", "实验目标", "前置条件", "操作步骤", "结果观察", "结果分析", "注意事项"),
            solver_hint="仅面向课程实验和授权环境；不要扩展成真实目标攻击指南。",
            priority=82,
        ),
        ProblemTemplate(
            id="cybersec_phenomenon_analysis",
            subject_id="cybersec_lab",
            name="网络安全实验现象分析题",
            problem_type="cybersec_phenomenon_analysis",
            aliases=("现象分析", "结果分析", "为什么", "抓包", "日志", "失败原因"),
            knowledge_keywords=("实验现象", "协议过程", "抓包分析", "日志分析", "故障排查"),
            steps=(
                "先复述现象并区分正常现象、异常现象和条件不足。",
                "定位涉及的协议/算法/实验步骤。",
                "从配置、密钥/证书、网络连通、权限、数据格式等角度分析原因。",
                "给出验证方法和修正建议。",
            ),
            output_sections=("题型判断", "现象复述", "原理定位", "原因分析", "验证方法", "结论", "注意事项"),
            solver_hint="优先解释课程实验现象，不提供未授权攻击的可执行操作链。",
            priority=84,
        ),
        ProblemTemplate(
            id="subject_general_problem",
            subject_id="C_program",
            name="课程通用题目辅导",
            problem_type="general_problem",
            aliases=("题目", "解题", "分析", "答案"),
            knowledge_keywords=("知识点", "题型", "步骤", "考试表达"),
            steps=(
                "识别题目目标和已知条件。",
                "定位考察知识点。",
                "按条件到结论的顺序分步解释。",
                "给出考试作答版本和易错点。",
            ),
            output_sections=("题型判断", "考点定位", "解题过程", "结论", "考试写法", "易错点"),
            solver_hint="题型不够明确时，先说明假设，再按最可能的课程题目处理。",
            priority=10,
        ),
        ProblemTemplate(
            id="os_general_problem",
            subject_id="operating_systems",
            name="操作系统通用题目辅导",
            problem_type="general_problem",
            aliases=("题目", "解题", "分析", "答案"),
            knowledge_keywords=("操作系统", "机制", "步骤", "考试表达"),
            steps=(
                "抽取题目条件和求解目标。",
                "定位对应章节、机制或算法。",
                "按机制流程或计算步骤展开。",
                "给出简洁考试作答版本。",
            ),
            output_sections=("题型判断", "考点定位", "解题过程", "结论", "考试写法", "易错点"),
            solver_hint="操作系统题优先讲清状态变化、算法规则或机制因果。",
            priority=10,
        ),
        ProblemTemplate(
            id="cybersec_general_problem",
            subject_id="cybersec_lab",
            name="网络安全实验通用题目辅导",
            problem_type="general_problem",
            aliases=("题目", "实验", "分析", "答案"),
            knowledge_keywords=("实验目标", "实验原理", "步骤", "现象分析"),
            steps=(
                "明确实验题目要求和授权环境。",
                "定位对应实验原理与步骤。",
                "结合题干现象或要求分步解释。",
                "给出实验报告式答案和注意事项。",
            ),
            output_sections=("题型判断", "实验原理", "解题过程", "实验结论", "注意事项", "类题训练方向"),
            solver_hint="回答必须限定在课程实验/授权环境中。",
            priority=10,
        ),
    )

    def __init__(
        self,
        llm_client: Any | None = None,
        *,
        question_bank_path: str | Path | None = None,
        question_bank_embedding_index_path: str | Path | None = None,
        question_bank_embed_enabled: bool | None = None,
        question_bank_embedder: Any | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.llm_analysis_struct = (
            llm_client.with_structured_output(TutoringProblemAnalysis)
            if llm_client is not None
            else None
        )
        self.question_bank_path = Path(question_bank_path or self.DEFAULT_QUESTION_BANK_PATH)
        self.question_bank_embedding_index_path = Path(
            question_bank_embedding_index_path
            or os.getenv("QUESTION_BANK_EMBED_INDEX_PATH", "").strip()
            or self.DEFAULT_QUESTION_BANK_EMBEDDING_INDEX_PATH
        )
        self._question_bank_cache: list[dict[str, Any]] | None = None
        self._question_bank_embedding_index_cache: dict[str, dict[str, Any]] | None = None
        self.question_bank_embed_enabled = (
            self._env_flag("QUESTION_BANK_EMBED_ENABLED", True)
            if question_bank_embed_enabled is None
            else bool(question_bank_embed_enabled)
        )
        self.question_bank_embed_model = (
            os.getenv("QUESTION_BANK_EMBED_MODEL", "text-embedding-3-small").strip()
            or "text-embedding-3-small"
        )
        self.question_bank_embed_top_k = self._safe_positive_int(
            os.getenv("QUESTION_BANK_EMBED_TOP_K", "8"),
            default=8,
        )
        self.question_bank_embed_batch_size = self._safe_positive_int(
            os.getenv("QUESTION_BANK_EMBED_BATCH_SIZE", "32"),
            default=32,
        )
        self.question_bank_embed_trigger_score = self._safe_positive_float(
            os.getenv("QUESTION_BANK_EMBED_TRIGGER_SCORE", "185"),
            default=185.0,
        )
        self.question_bank_embedder = question_bank_embedder

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _safe_positive_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _safe_positive_float(value: Any, *, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).lower()

    @staticmethod
    def _short_text(text: str, *, max_len: int = 4000) -> str:
        value = str(text or "").strip()
        if len(value) <= max_len:
            return value
        return value[:max_len].rstrip() + "\n...[已截断]"

    @staticmethod
    def _normalize_exception_message(exc: Exception) -> str:
        message = str(exc).strip()
        return message or type(exc).__name__

    @staticmethod
    def _model_to_dict(obj: Any) -> dict[str, Any]:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return dict(obj)

    def load_question_bank(self) -> list[dict[str, Any]]:
        if self._question_bank_cache is not None:
            return list(self._question_bank_cache)
        path = self.question_bank_path
        if not path.exists():
            self._question_bank_cache = []
            return []

        items: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"题库 JSONL 第 {line_no} 行格式错误: {exc}") from exc
            if isinstance(item, dict):
                items.append(item)
        self._question_bank_cache = items
        return list(items)

    @staticmethod
    def _similarity_terms(text: str) -> set[str]:
        raw = str(text or "").lower()
        terms = set(re.findall(r"[a-z_][a-z0-9_]*|\d+", raw))
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", raw))
        terms.update(cjk[i : i + 2] for i in range(max(0, len(cjk) - 1)))
        return {term for term in terms if term}

    @staticmethod
    def _question_bank_public_item(item: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "id": str(item.get("id", "")),
            "subject_id": str(item.get("subject_id", "")),
            "problem_type": str(item.get("problem_type", "")),
            "difficulty": str(item.get("difficulty", "")),
            "knowledge_points": list(item.get("knowledge_points", []) or []),
            "question": str(item.get("question", "")),
            "score": round(float(score), 4),
        }

    @classmethod
    def build_question_bank_embedding_text(
        cls,
        *,
        subject_id: str,
        problem_type: str,
        knowledge_points: list[str] | tuple[str, ...] | set[str],
        question: str,
    ) -> str:
        subject_label = cls.SUBJECT_LABELS.get(subject_id, subject_id or "未知学科")
        points = " ".join(str(point).strip() for point in knowledge_points if str(point).strip())
        return "\n".join(
            [
                f"学科: {subject_label}",
                f"题型: {str(problem_type or '').strip()}",
                f"知识点: {points}",
                f"题目: {str(question or '').strip()}",
            ]
        ).strip()

    def load_question_bank_embedding_index(self) -> dict[str, dict[str, Any]]:
        if self._question_bank_embedding_index_cache is not None:
            return dict(self._question_bank_embedding_index_cache)
        path = self.question_bank_embedding_index_path
        if not self.question_bank_embed_enabled or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        index: dict[str, dict[str, Any]] = {}
        items = payload.get("items", []) if isinstance(payload, dict) else []
        for item in list(items or []):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "") or "").strip()
            vector = self._coerce_embedding_vector(item.get("embedding"))
            if not item_id or not vector:
                continue
            index[item_id] = {
                "embedding": vector,
                "subject_id": str(item.get("subject_id", "") or ""),
                "problem_type": str(item.get("problem_type", "") or ""),
                "knowledge_points": [
                    str(point).strip()
                    for point in list(item.get("knowledge_points", []) or [])
                    if str(point).strip()
                ],
            }
        self._question_bank_embedding_index_cache = index
        return dict(index)

    @staticmethod
    def _coerce_embedding_vector(value: Any) -> list[float]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or not value:
            return []
        vector: list[float] = []
        for item in value:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                return []
            if math.isnan(numeric) or math.isinf(numeric):
                return []
            vector.append(numeric)
        return vector

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))

    async def _embed_question_bank_texts(
        self,
        texts: list[str],
        *,
        raise_on_error: bool = False,
    ) -> list[list[float]]:
        clean_texts = [str(text or "").strip() for text in list(texts or []) if str(text or "").strip()]
        if not clean_texts:
            return []
        try:
            if self.question_bank_embedder is not None:
                raw_vectors = self.question_bank_embedder(clean_texts)
                if asyncio.iscoroutine(raw_vectors):
                    raw_vectors = await raw_vectors
            else:
                from lightrag.llm.openai import openai_embed

                raw_vectors = await openai_embed.func(
                    clean_texts,
                    model=self.question_bank_embed_model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    max_token_size=0,
                )
        except Exception:
            if raise_on_error:
                raise
            return []
        if hasattr(raw_vectors, "tolist"):
            raw_vectors = raw_vectors.tolist()
        elif isinstance(raw_vectors, tuple):
            raw_vectors = list(raw_vectors)
        if not isinstance(raw_vectors, list):
            if raise_on_error:
                raise ValueError("题库 embedding 返回了无法解析的向量列表")
            return []
        vectors: list[list[float]] = []
        for item in raw_vectors:
            vector = self._coerce_embedding_vector(item)
            if not vector:
                if raise_on_error:
                    raise ValueError("题库 embedding 返回了非法向量")
                return []
            vectors.append(vector)
        if len(vectors) != len(clean_texts):
            if raise_on_error:
                raise ValueError("题库 embedding 返回数量与输入不一致")
            return []
        return vectors

    async def _embed_question_bank_query(self, text: str) -> list[float]:
        vectors = await self._embed_question_bank_texts([text])
        return vectors[0] if vectors else []

    def _build_question_bank_query_context(
        self,
        *,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        user_question: str,
    ) -> dict[str, Any]:
        subject_id = analysis.subject_id if analysis.subject_id != "unknown" else template.subject_id
        problem_type = str(analysis.problem_type or template.problem_type).strip()
        knowledge_points = {
            str(point).strip().lower()
            for point in (analysis.knowledge_points or template.knowledge_keywords)
            if str(point).strip()
        }
        return {
            "subject_id": subject_id,
            "problem_type": problem_type,
            "knowledge_points": knowledge_points,
            "query_terms": self._similarity_terms(
                " ".join(
                    [
                        user_question,
                        problem_type,
                        " ".join(sorted(knowledge_points)),
                        template.name,
                    ]
                )
            ),
            "normalized_user_question": self._normalize_text(user_question),
            "embedding_text": self.build_question_bank_embedding_text(
                subject_id=subject_id,
                problem_type=problem_type,
                knowledge_points=sorted(knowledge_points),
                question=user_question,
            ),
        }

    def _score_question_bank_lexical_candidates(
        self,
        *,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        user_question: str,
    ) -> list[tuple[float, dict[str, Any]]]:
        bank = self.load_question_bank()
        if not bank:
            return []

        context = self._build_question_bank_query_context(
            analysis=analysis,
            template=template,
            user_question=user_question,
        )
        subject_id = str(context["subject_id"])
        problem_type = str(context["problem_type"])
        knowledge_points = set(context["knowledge_points"])
        query_terms = set(context["query_terms"])
        normalized_user_question = str(context["normalized_user_question"])

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in bank:
            item_subject = str(item.get("subject_id", "") or "")
            if item_subject != subject_id:
                continue
            item_question = str(item.get("question", "") or "")
            if normalized_user_question and self._normalize_text(item_question) == normalized_user_question:
                continue

            score = 100.0
            item_problem_type = str(item.get("problem_type", "") or "")
            if item_problem_type == problem_type:
                score += 60.0
            elif item_problem_type == template.problem_type:
                score += 35.0

            item_points = {
                str(point).strip().lower()
                for point in (item.get("knowledge_points", []) or [])
                if str(point).strip()
            }
            overlap_points = knowledge_points & item_points
            score += len(overlap_points) * 18.0

            item_terms = self._similarity_terms(
                " ".join(
                    [
                        item_question,
                        item_problem_type,
                        " ".join(item_points),
                    ]
                )
            )
            if query_terms and item_terms:
                overlap_terms = query_terms & item_terms
                score += min(40.0, len(overlap_terms) * 2.0)
                score += 20.0 * (len(overlap_terms) / max(1, len(query_terms)))

            difficulty = str(item.get("difficulty", "") or "").lower()
            if difficulty == "medium":
                score += 2.0

            scored.append((score, item))

        scored.sort(key=lambda row: (-row[0], str(row[1].get("id", ""))))
        return scored

    def build_few_shot_examples(
        self,
        recommendations: list[dict[str, Any]],
        *,
        limit: int = 2,
    ) -> list[dict[str, Any]]:
        if not recommendations:
            return []
        by_id = {
            str(item.get("id", "") or ""): item
            for item in self.load_question_bank()
            if str(item.get("id", "") or "")
        }
        examples: list[dict[str, Any]] = []
        for rec in recommendations[: max(0, int(limit))]:
            item = by_id.get(str(rec.get("id", "") or ""))
            if not item:
                continue
            examples.append(
                {
                    "id": str(item.get("id", "")),
                    "subject_id": str(item.get("subject_id", "")),
                    "problem_type": str(item.get("problem_type", "")),
                    "difficulty": str(item.get("difficulty", "")),
                    "question": self._short_text(str(item.get("question", "") or ""), max_len=700),
                    "answer": self._short_text(str(item.get("answer", "") or ""), max_len=420),
                    "solution_steps": [
                        self._short_text(str(step), max_len=180)
                        for step in list(item.get("solution_steps", []) or [])[:6]
                    ],
                    "common_mistakes": [
                        self._short_text(str(mistake), max_len=160)
                        for mistake in list(item.get("common_mistakes", []) or [])[:3]
                    ],
                    "use_policy": "仅模仿解题结构、步骤粒度和易错点写法；不要照搬示例题的具体数值或结论。",
                }
            )
        return examples

    def _unique_short_texts(
        self,
        values: list[Any] | tuple[Any, ...],
        *,
        max_items: int,
        max_len: int,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in list(values or []):
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            if not text:
                continue
            short = self._short_text(text, max_len=max_len)
            if short in seen:
                continue
            seen.add(short)
            out.append(short)
            if len(out) >= max(1, int(max_items)):
                break
        return out

    def _collect_common_mistakes(
        self,
        few_shot_examples: list[dict[str, Any]],
        *,
        max_items: int = 4,
    ) -> list[str]:
        mistakes: list[Any] = []
        for item in list(few_shot_examples or []):
            mistakes.extend(list(item.get("common_mistakes", []) or []))
        return self._unique_short_texts(
            mistakes,
            max_items=max_items,
            max_len=160,
        )

    def _solver_display_name(self, solver: str) -> str:
        return {
            "page_replacement": "页面置换规则求解",
            "cpu_scheduling": "进程调度规则求解",
            "banker": "银行家算法规则求解",
            "pv_sync": "PV 同步结构求解",
            "diffie_hellman": "DH 规则求解",
            "none": "未命中规则求解器",
        }.get(str(solver or "").strip(), str(solver or "").strip() or "规则求解")

    def _build_solver_summary(self, solver_result: dict[str, Any]) -> str:
        status = str(solver_result.get("status", "") or "").strip().lower()
        message = self._short_text(str(solver_result.get("message", "") or ""), max_len=180)
        if status != "success":
            return message or ("规则求解失败。" if status == "failure" else "未运行规则求解器。")

        solver = str(solver_result.get("solver", "") or "").strip()
        result = solver_result.get("result") or {}
        if not isinstance(result, dict):
            return message or "规则求解成功。"

        if solver == "page_replacement":
            algorithm = str(result.get("algorithm", "") or "").strip()
            faults = result.get("faults")
            hits = result.get("hits")
            return f"{algorithm} 计算完成：缺页 {faults} 次，命中 {hits} 次。".strip()
        if solver == "cpu_scheduling":
            algorithm = str(result.get("algorithm", "") or "").strip()
            avg_turnaround = result.get("average_turnaround")
            avg_waiting = result.get("average_waiting")
            return (
                f"{algorithm} 调度完成：平均周转时间 {avg_turnaround}，"
                f"平均等待时间 {avg_waiting}。"
            ).strip()
        if solver == "banker":
            if "grantable" in result:
                grantable = bool(result.get("grantable"))
                sequence = " -> ".join(result.get("safe_sequence", []) or [])
                conclusion = "请求可以分配" if grantable else "请求不能分配"
                if sequence:
                    return f"{conclusion}；试分配后的安全序列为 {sequence}。"
                return f"{conclusion}。"
            if "safe" in result:
                safe = bool(result.get("safe"))
                sequence = " -> ".join(result.get("safe_sequence", []) or [])
                conclusion = "系统处于安全状态" if safe else "系统处于不安全状态"
                if sequence:
                    return f"{conclusion}；安全序列为 {sequence}。"
                return f"{conclusion}。"
            if "need" in result:
                return f"Need 计算完成：{result.get('need')}。"
        if solver == "pv_sync":
            scenario = str(result.get("scenario", "") or "").strip()
            semaphore_count = len(list(result.get("semaphores", []) or []))
            return f"已生成 {scenario or 'PV'} 场景的信号量设计，共 {semaphore_count} 个关键信号量。"
        if solver == "diffie_hellman":
            shared_key = result.get("shared_key")
            return f"DH 推导完成：共享密钥为 {shared_key}。"
        return message or "规则求解成功。"

    def _build_retrieval_summary(self, retrieval: dict[str, Any]) -> str:
        status = str(retrieval.get("status", "") or "").strip().lower()
        answer = self._short_text(str(retrieval.get("answer", "") or ""), max_len=180)
        message = self._short_text(str(retrieval.get("message", "") or ""), max_len=120)
        if status == "success" and answer:
            return answer
        return answer or message or "未检索到明确课程依据。"

    def build_learning_outline(
        self,
        *,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        recommendations: list[dict[str, Any]] | None = None,
        few_shot_examples: list[dict[str, Any]] | None = None,
        solver_result: dict[str, Any] | None = None,
        retrieval: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        solver = dict(solver_result or {})
        retrieval_data = dict(retrieval or {})
        recommended_steps_source = (
            list(solver.get("steps", []) or [])
            if str(solver.get("status", "") or "").strip().lower() == "success"
            else list(template.steps or [])
        )
        similar_questions: list[dict[str, Any]] = []
        for item in list(recommendations or [])[:3]:
            similar_questions.append(
                {
                    "id": str(item.get("id", "") or "").strip(),
                    "difficulty": str(item.get("difficulty", "") or "").strip(),
                    "question": self._short_text(
                        str(item.get("question", "") or ""),
                        max_len=120,
                    ),
                    "score": round(float(item.get("score", 0.0) or 0.0), 4),
                }
            )

        return {
            "kind": "problem_tutoring",
            "subject_id": analysis.subject_id,
            "subject_label": self.SUBJECT_LABELS.get(analysis.subject_id, analysis.subject_id),
            "problem_type": str(analysis.problem_type or template.problem_type),
            "problem_type_label": template.name,
            "confidence": round(float(analysis.confidence), 4),
            "target": self._short_text(str(analysis.target or ""), max_len=140),
            "knowledge_points": self._unique_short_texts(
                list(analysis.knowledge_points or template.knowledge_keywords),
                max_items=6,
                max_len=48,
            ),
            "conditions": self._unique_short_texts(
                list(analysis.extracted_conditions or []),
                max_items=6,
                max_len=120,
            ),
            "recommended_steps": self._unique_short_texts(
                recommended_steps_source,
                max_items=6,
                max_len=140,
            ),
            "solver": {
                "status": str(solver.get("status", "") or "").strip(),
                "name": self._solver_display_name(str(solver.get("solver", "") or "")),
                "summary": self._build_solver_summary(solver),
                "message": self._short_text(str(solver.get("message", "") or ""), max_len=160),
            },
            "retrieval": {
                "status": str(retrieval_data.get("status", "") or "").strip(),
                "summary": self._build_retrieval_summary(retrieval_data),
            },
            "common_mistakes": self._collect_common_mistakes(
                list(few_shot_examples or []),
                max_items=4,
            ),
            "similar_questions": similar_questions,
        }

    def recommend_similar_questions(
        self,
        *,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        user_question: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        scored = self._score_question_bank_lexical_candidates(
            analysis=analysis,
            template=template,
            user_question=user_question,
        )
        return [
            self._question_bank_public_item(item, score)
            for score, item in scored[: max(0, int(limit))]
        ]

    def _should_use_embedding_recommendations(
        self,
        lexical_candidates: list[tuple[float, dict[str, Any]]],
        *,
        limit: int,
    ) -> bool:
        if not self.question_bank_embed_enabled:
            return False
        if not self.load_question_bank_embedding_index():
            return False
        if len(lexical_candidates) < max(1, int(limit)):
            return True
        top_score = lexical_candidates[0][0] if lexical_candidates else 0.0
        return float(top_score) < self.question_bank_embed_trigger_score

    async def recommend_similar_questions_by_embedding(
        self,
        *,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        user_question: str,
        limit: int = 8,
    ) -> list[tuple[float, dict[str, Any]]]:
        index = self.load_question_bank_embedding_index()
        if not index:
            return []

        context = self._build_question_bank_query_context(
            analysis=analysis,
            template=template,
            user_question=user_question,
        )
        query_vector = await self._embed_question_bank_query(str(context["embedding_text"]))
        if not query_vector:
            return []

        subject_id = str(context["subject_id"])
        normalized_user_question = str(context["normalized_user_question"])
        by_id = {
            str(item.get("id", "") or ""): item
            for item in self.load_question_bank()
            if str(item.get("id", "") or "")
        }
        scored: list[tuple[float, dict[str, Any]]] = []
        for item_id, entry in index.items():
            item = by_id.get(item_id)
            if not item:
                continue
            if str(item.get("subject_id", "") or "") != subject_id:
                continue
            item_question = str(item.get("question", "") or "")
            if normalized_user_question and self._normalize_text(item_question) == normalized_user_question:
                continue
            similarity = self._cosine_similarity(query_vector, list(entry.get("embedding", []) or []))
            if similarity <= 0.0:
                continue
            scored.append((similarity, item))

        scored.sort(key=lambda row: (-row[0], str(row[1].get("id", ""))))
        return scored[: max(0, int(limit))]

    def _merge_question_bank_recommendations(
        self,
        *,
        lexical_candidates: list[tuple[float, dict[str, Any]]],
        embedding_candidates: list[tuple[float, dict[str, Any]]],
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        user_question: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        context = self._build_question_bank_query_context(
            analysis=analysis,
            template=template,
            user_question=user_question,
        )
        problem_type = str(context["problem_type"])
        knowledge_points = set(context["knowledge_points"])

        lexical_map = {
            str(item.get("id", "") or ""): float(score) for score, item in lexical_candidates
        }
        embedding_map = {
            str(item.get("id", "") or ""): float(score) for score, item in embedding_candidates
        }
        lexical_max = max(lexical_map.values(), default=1.0)
        embedding_max = max(embedding_map.values(), default=1.0)

        merged_items: dict[str, dict[str, Any]] = {}
        for score, item in lexical_candidates[: max(limit * 4, 8)]:
            del score
            item_id = str(item.get("id", "") or "")
            if item_id:
                merged_items[item_id] = item
        for score, item in embedding_candidates[: max(limit * 4, 8)]:
            del score
            item_id = str(item.get("id", "") or "")
            if item_id:
                merged_items[item_id] = item

        ranked: list[tuple[float, dict[str, Any]]] = []
        for item_id, item in merged_items.items():
            item_problem_type = str(item.get("problem_type", "") or "")
            item_points = {
                str(point).strip().lower()
                for point in (item.get("knowledge_points", []) or [])
                if str(point).strip()
            }
            point_overlap = len(knowledge_points & item_points) / max(1, len(knowledge_points))
            type_bonus = 0.15 if item_problem_type == problem_type else 0.08 if item_problem_type == template.problem_type else 0.0
            lexical_score = lexical_map.get(item_id, 0.0) / max(1.0, lexical_max)
            embedding_score = embedding_map.get(item_id, 0.0) / max(1.0, embedding_max)
            final_score = (
                lexical_score * 0.45
                + embedding_score * 0.30
                + type_bonus
                + min(0.10, point_overlap * 0.10)
            )
            ranked.append((final_score, item))

        ranked.sort(key=lambda row: (-row[0], str(row[1].get("id", ""))))
        return [
            self._question_bank_public_item(item, score * 100.0)
            for score, item in ranked[: max(0, int(limit))]
        ]

    async def recommend_similar_questions_hybrid(
        self,
        *,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        user_question: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        lexical_candidates = self._score_question_bank_lexical_candidates(
            analysis=analysis,
            template=template,
            user_question=user_question,
        )
        lexical_results = [
            self._question_bank_public_item(item, score)
            for score, item in lexical_candidates[: max(0, int(limit))]
        ]
        if not self._should_use_embedding_recommendations(lexical_candidates, limit=limit):
            return lexical_results
        embedding_candidates = await self.recommend_similar_questions_by_embedding(
            analysis=analysis,
            template=template,
            user_question=user_question,
            limit=self.question_bank_embed_top_k,
        )
        if not embedding_candidates:
            return lexical_results
        return self._merge_question_bank_recommendations(
            lexical_candidates=lexical_candidates,
            embedding_candidates=embedding_candidates,
            analysis=analysis,
            template=template,
            user_question=user_question,
            limit=limit,
        )

    async def build_question_bank_embedding_index(
        self,
        *,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        bank = self.load_question_bank()
        if not bank:
            raise ValueError("题库为空，无法构建 embedding 索引")

        records: list[dict[str, Any]] = []
        batch_size = max(1, self.question_bank_embed_batch_size)
        for start in range(0, len(bank), batch_size):
            batch = bank[start : start + batch_size]
            batch_texts = [
                self.build_question_bank_embedding_text(
                    subject_id=str(item.get("subject_id", "") or ""),
                    problem_type=str(item.get("problem_type", "") or ""),
                    knowledge_points=list(item.get("knowledge_points", []) or []),
                    question=str(item.get("question", "") or ""),
                )
                for item in batch
            ]
            vectors = await self._embed_question_bank_texts(batch_texts, raise_on_error=True)
            for item, vector, text in zip(batch, vectors, batch_texts):
                records.append(
                    {
                        "id": str(item.get("id", "") or ""),
                        "subject_id": str(item.get("subject_id", "") or ""),
                        "problem_type": str(item.get("problem_type", "") or ""),
                        "knowledge_points": [
                            str(point).strip()
                            for point in list(item.get("knowledge_points", []) or [])
                            if str(point).strip()
                        ],
                        "text": text,
                        "embedding": vector,
                    }
                )

        target_path = Path(output_path or self.question_bank_embedding_index_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.question_bank_embed_model,
            "created_at": datetime.now(UTC).isoformat(),
            "item_count": len(records),
            "items": records,
        }
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        self._question_bank_embedding_index_cache = None
        return {
            "path": str(target_path),
            "item_count": len(records),
            "model": self.question_bank_embed_model,
        }

    @staticmethod
    def _solver_skipped(solver: str, message: str, **details: Any) -> dict[str, Any]:
        return {
            "status": "skipped",
            "solver": solver,
            "message": message,
            "details": details,
        }

    @staticmethod
    def _solver_success(solver: str, result: dict[str, Any], steps: list[str]) -> dict[str, Any]:
        return {
            "status": "success",
            "solver": solver,
            "message": "规则求解成功",
            "result": result,
            "steps": steps,
        }

    @staticmethod
    def _solver_failure(solver: str, exc: Exception) -> dict[str, Any]:
        return {
            "status": "failure",
            "solver": solver,
            "message": str(exc).strip() or type(exc).__name__,
            "failure_reason": type(exc).__name__,
        }

    @staticmethod
    def _extract_ints(text: str) -> list[int]:
        return [int(item) for item in re.findall(r"-?\d+", str(text or ""))]

    @staticmethod
    def _extract_labeled_int(text: str, patterns: tuple[str, ...]) -> int | None:
        for pattern in patterns:
            match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _parse_vector_text(text: str) -> tuple[int, ...] | None:
        values = ProblemTutoringService._extract_ints(text)
        return tuple(values) if values else None

    @staticmethod
    def _vector_leq(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
        return len(left) == len(right) and all(a <= b for a, b in zip(left, right))

    @staticmethod
    def _vector_add(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(a + b for a, b in zip(left, right))

    @staticmethod
    def _vector_sub(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(a - b for a, b in zip(left, right))

    @staticmethod
    def _process_sort_key(name: str) -> tuple[int, str]:
        match = re.search(r"\d+", name)
        return (int(match.group(0)) if match else 10**9, name)

    def solve_deterministic(
        self,
        *,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        user_question: str,
    ) -> dict[str, Any]:
        problem_type = str(analysis.problem_type or template.problem_type)
        text = str(user_question or "")
        try:
            if problem_type == "os_page_replacement":
                return self._solve_page_replacement(text)
            if problem_type == "os_cpu_scheduling":
                return self._solve_cpu_scheduling(text)
            if problem_type == "os_banker":
                return self._solve_banker(text)
            if problem_type == "os_pv_sync":
                return self._solve_pv_sync(text)
            if self._looks_like_dh_problem(text, analysis=analysis, template=template):
                return self._solve_diffie_hellman(text)
            return self._solver_skipped(
                "none",
                "当前题型暂未配置确定性规则求解器，回退到模板、课程检索和 LLM 推理。",
                problem_type=problem_type,
            )
        except Exception as exc:
            return self._solver_failure(problem_type, exc)

    @staticmethod
    def _pick_page_algorithm(text: str) -> str | None:
        normalized = str(text or "").upper()
        if "FIFO" in normalized or "先进先出" in text:
            return "FIFO"
        if "LRU" in normalized or "最近最久未使用" in text:
            return "LRU"
        if "OPT" in normalized or "最佳置换" in text or "最佳页面置换" in text:
            return "OPT"
        if "CLOCK" in normalized or "时钟" in text:
            return "CLOCK"
        return None

    @staticmethod
    def _extract_page_sequence(text: str) -> list[int]:
        patterns = (
            r"(?:页面访问序列|访问序列|页面序列|页号序列|引用串|访问串)\s*(?:为|是|=|:|：)?\s*([0-9,\s，、]+)",
            r"(?:sequence|reference string)\s*(?:=|:)?\s*([0-9,\s,]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                values = ProblemTutoringService._extract_ints(match.group(1))
                if len(values) >= 2:
                    return values
        return []

    @staticmethod
    def _extract_page_frame_count(text: str) -> int | None:
        return ProblemTutoringService._extract_labeled_int(
            text,
            (
                r"(?:页框数|页框|物理块数|内存块数)\s*(?:为|是|=|:|：)?\s*(\d+)",
                r"(\d+)\s*(?:个)?(?:页框|物理块|内存块)",
                r"(?:frames?|frame count)\s*(?:=|:)?\s*(\d+)",
            ),
        )

    def _solve_page_replacement(self, text: str) -> dict[str, Any]:
        algorithm = self._pick_page_algorithm(text)
        if algorithm is None:
            return self._solver_skipped("page_replacement", "未识别页面置换算法，需要 FIFO、LRU、OPT 或 Clock。")

        sequence = self._extract_page_sequence(text)
        frames_count = self._extract_page_frame_count(text)
        if not sequence or frames_count is None:
            return self._solver_skipped(
                "page_replacement",
                "页面访问序列或页框数不完整，不能确定性计算。",
                sequence=sequence,
                frames=frames_count,
            )
        if frames_count <= 0:
            return self._solver_skipped("page_replacement", "页框数必须大于 0。", frames=frames_count)

        memory: list[int] = []
        fifo_queue: list[int] = []
        last_used: dict[int, int] = {}
        reference_bits: list[int] = []
        clock_hand = 0
        trace: list[dict[str, Any]] = []
        faults = 0
        hits = 0
        for index, page in enumerate(sequence):
            before = list(memory)
            bits_before = list(reference_bits)
            hand_before = clock_hand
            victim: int | None = None
            if page in memory:
                hits += 1
                event = "hit"
                last_used[page] = index
                if algorithm == "CLOCK":
                    reference_bits[memory.index(page)] = 1
            else:
                faults += 1
                event = "fault"
                if len(memory) < frames_count:
                    memory.append(page)
                    if algorithm == "FIFO":
                        fifo_queue.append(page)
                    if algorithm == "CLOCK":
                        reference_bits.append(1)
                else:
                    if algorithm == "FIFO":
                        victim = fifo_queue.pop(0)
                    elif algorithm == "LRU":
                        victim = min(memory, key=lambda item: last_used.get(item, -1))
                    elif algorithm == "OPT":
                        future = sequence[index + 1 :]
                        victim = max(
                            memory,
                            key=lambda item: future.index(item) if item in future else 10**9,
                        )
                    else:
                        while reference_bits[clock_hand] == 1:
                            reference_bits[clock_hand] = 0
                            clock_hand = (clock_hand + 1) % frames_count
                        victim = memory[clock_hand]
                    replace_index = clock_hand if algorithm == "CLOCK" else memory.index(victim)
                    memory[replace_index] = page
                    if algorithm == "FIFO":
                        fifo_queue.append(page)
                    if algorithm == "CLOCK":
                        reference_bits[replace_index] = 1
                        clock_hand = (replace_index + 1) % frames_count
                last_used[page] = index

            trace.append(
                {
                    "step": index + 1,
                    "page": page,
                    "event": event,
                    "frames_before": before,
                    "evicted": victim,
                    "frames_after": list(memory),
                    "reference_bits_before": bits_before if algorithm == "CLOCK" else None,
                    "reference_bits_after": list(reference_bits) if algorithm == "CLOCK" else None,
                    "clock_hand_before": hand_before if algorithm == "CLOCK" else None,
                    "clock_hand_after": clock_hand if algorithm == "CLOCK" else None,
                }
            )

        total = len(sequence)
        result = {
            "algorithm": algorithm,
            "frames": frames_count,
            "sequence": sequence,
            "faults": faults,
            "hits": hits,
            "fault_rate": round(faults / total, 4) if total else 0,
            "hit_rate": round(hits / total, 4) if total else 0,
            "trace": trace,
        }
        steps = [
            f"识别算法为 {algorithm}，页框数为 {frames_count}，访问序列长度为 {total}。",
            "逐次访问页面：命中不增加缺页次数；缺页且页框未满时直接装入。",
            f"页框满时按 {algorithm} 规则选择被置换页面。",
            f"统计得到缺页 {faults} 次、命中 {hits} 次，缺页率 {result['fault_rate']}。",
        ]
        return self._solver_success("page_replacement", result, steps)

    @staticmethod
    def _parse_process_literal(text: str) -> list[tuple[str, int, int]]:
        match = re.search(r"进程\s*(\[[^\]]+\])", text)
        if not match:
            return []
        try:
            value = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            return []
        processes: list[tuple[str, int, int]] = []
        if not isinstance(value, list):
            return []
        for item in value:
            if not isinstance(item, tuple) or len(item) < 3:
                continue
            name, arrival, service = item[:3]
            processes.append((str(name), int(arrival), int(service)))
        return processes

    @staticmethod
    def _parse_process_chinese(text: str) -> list[tuple[str, int, int]]:
        processes: dict[str, tuple[str, int, int]] = {}
        for segment in re.split(r"[;；。\n]", text):
            match = re.search(
                r"(P\d+).*?(?:到达时间|到达)\s*(?:为|是|=|:|：)?\s*(\d+).*?"
                r"(?:服务时间|运行时间|执行时间|服务)\s*(?:为|是|=|:|：)?\s*(\d+)",
                segment,
                flags=re.IGNORECASE,
            )
            if match:
                name = match.group(1).upper()
                processes[name] = (name, int(match.group(2)), int(match.group(3)))

        for name, arrival, service in re.findall(
            r"(P\d+)\s*[（(]\s*(\d+)\s*[,，]\s*(\d+)\s*[）)]",
            text,
            flags=re.IGNORECASE,
        ):
            key = name.upper()
            processes[key] = (key, int(arrival), int(service))
        return [processes[key] for key in sorted(processes, key=ProblemTutoringService._process_sort_key)]

    @staticmethod
    def _parse_scheduling_processes(text: str) -> list[tuple[str, int, int]]:
        processes = ProblemTutoringService._parse_process_literal(text)
        if processes:
            return processes
        return ProblemTutoringService._parse_process_chinese(text)

    @staticmethod
    def _pick_scheduling_algorithm(text: str) -> str | None:
        normalized = str(text or "").upper()
        if "SRTF" in normalized or "最短剩余时间" in text or "抢占式短作业" in text:
            return "SRTF"
        if "RR" in normalized or "时间片" in text or "轮转" in text:
            return "RR"
        if "SJF" in normalized or "短作业" in text:
            return "SJF"
        if "FCFS" in normalized or "先来先服务" in text:
            return "FCFS"
        return None

    @staticmethod
    def _extract_quantum(text: str) -> int | None:
        return ProblemTutoringService._extract_labeled_int(
            text,
            (
                r"(?:时间片|q|quantum)\s*(?:为|是|=|:|：)?\s*(\d+)",
                r"(\d+)\s*(?:个)?时间片",
            ),
        )

    @staticmethod
    def _metrics_from_completion(
        processes: list[tuple[str, int, int]],
        completion: dict[str, int],
        first_start: dict[str, int] | None = None,
    ) -> dict[str, dict[str, float]]:
        by_name = {name: (arrival, service) for name, arrival, service in processes}
        metrics: dict[str, dict[str, float]] = {}
        for name in sorted(completion, key=ProblemTutoringService._process_sort_key):
            arrival, service = by_name[name]
            turnaround = completion[name] - arrival
            waiting = turnaround - service
            metrics[name] = {
                "completion": completion[name],
                "turnaround": turnaround,
                "waiting": waiting,
                "weighted_turnaround": round(turnaround / service, 4) if service else 0,
            }
            if first_start and name in first_start:
                metrics[name]["start"] = first_start[name]
        return metrics

    def _solve_cpu_scheduling(self, text: str) -> dict[str, Any]:
        algorithm = self._pick_scheduling_algorithm(text)
        if algorithm is None:
            return self._solver_skipped("cpu_scheduling", "未识别调度算法，需要 FCFS、SJF、SRTF 或 RR。")
        processes = self._parse_scheduling_processes(text)
        if not processes:
            return self._solver_skipped("cpu_scheduling", "未抽取到进程到达时间和服务时间。")
        if any(service <= 0 for _name, _arrival, service in processes):
            return self._solver_skipped("cpu_scheduling", "服务时间必须大于 0。")

        processes = sorted(processes, key=lambda item: (item[1], self._process_sort_key(item[0])))
        timeline: list[dict[str, Any]] = []
        completion: dict[str, int] = {}
        first_start: dict[str, int] = {}

        if algorithm == "FCFS":
            time_now = 0
            for name, arrival, service in processes:
                start = max(time_now, arrival)
                finish = start + service
                timeline.append({"process": name, "start": start, "end": finish})
                first_start[name] = start
                completion[name] = finish
                time_now = finish
        elif algorithm == "SJF":
            remaining = list(processes)
            time_now = 0
            while remaining:
                ready = [item for item in remaining if item[1] <= time_now]
                if not ready:
                    time_now = min(item[1] for item in remaining)
                    ready = [item for item in remaining if item[1] <= time_now]
                name, arrival, service = min(
                    ready,
                    key=lambda item: (item[2], item[1], self._process_sort_key(item[0])),
                )
                remaining.remove((name, arrival, service))
                start = time_now
                finish = start + service
                timeline.append({"process": name, "start": start, "end": finish})
                first_start[name] = start
                completion[name] = finish
                time_now = finish
        elif algorithm == "RR":
            quantum = self._extract_quantum(text)
            if quantum is None or quantum <= 0:
                return self._solver_skipped("cpu_scheduling", "RR 调度缺少有效时间片 q。", quantum=quantum)

            arrivals = sorted(processes, key=lambda item: (item[1], self._process_sort_key(item[0])))
            remaining_time = {name: service for name, _arrival, service in arrivals}
            queue: list[str] = []
            index = 0
            time_now = 0
            while len(completion) < len(arrivals):
                while index < len(arrivals) and arrivals[index][1] <= time_now:
                    queue.append(arrivals[index][0])
                    index += 1
                if not queue:
                    if index >= len(arrivals):
                        break
                    time_now = max(time_now, arrivals[index][1])
                    continue
                name = queue.pop(0)
                run_time = min(quantum, remaining_time[name])
                start = time_now
                end = start + run_time
                first_start.setdefault(name, start)
                timeline.append({"process": name, "start": start, "end": end})
                time_now = end
                remaining_time[name] -= run_time
                while index < len(arrivals) and arrivals[index][1] <= time_now:
                    queue.append(arrivals[index][0])
                    index += 1
                if remaining_time[name] > 0:
                    queue.append(name)
                else:
                    completion[name] = time_now
        else:
            arrivals = sorted(processes, key=lambda item: (item[1], self._process_sort_key(item[0])))
            remaining_time = {name: service for name, _arrival, service in arrivals}
            by_name = {name: (arrival, service) for name, arrival, service in arrivals}
            time_now = min(arrival for _name, arrival, _service in arrivals)
            while len(completion) < len(arrivals):
                ready = [
                    name
                    for name, (arrival, _service) in by_name.items()
                    if arrival <= time_now and name not in completion and remaining_time[name] > 0
                ]
                if not ready:
                    future_arrivals = [
                        arrival
                        for name, (arrival, _service) in by_name.items()
                        if name not in completion and remaining_time[name] > 0 and arrival > time_now
                    ]
                    if not future_arrivals:
                        break
                    time_now = min(future_arrivals)
                    continue

                name = min(
                    ready,
                    key=lambda item: (
                        remaining_time[item],
                        by_name[item][0],
                        self._process_sort_key(item),
                    ),
                )
                next_arrivals = [
                    arrival
                    for other, (arrival, _service) in by_name.items()
                    if other not in completion
                    and other != name
                    and remaining_time[other] > 0
                    and arrival > time_now
                ]
                next_arrival = min(next_arrivals) if next_arrivals else None
                finish_time = time_now + remaining_time[name]
                end = min(finish_time, next_arrival) if next_arrival is not None else finish_time
                if end <= time_now:
                    end = time_now + 1
                first_start.setdefault(name, time_now)
                if timeline and timeline[-1]["process"] == name and timeline[-1]["end"] == time_now:
                    timeline[-1]["end"] = end
                else:
                    timeline.append({"process": name, "start": time_now, "end": end})
                remaining_time[name] -= end - time_now
                time_now = end
                if remaining_time[name] == 0:
                    completion[name] = time_now

        metrics = self._metrics_from_completion(processes, completion, first_start=first_start)
        result = {
            "algorithm": algorithm,
            "processes": [
                {"name": name, "arrival": arrival, "service": service}
                for name, arrival, service in processes
            ],
            "timeline": timeline,
            "metrics": metrics,
            "average_turnaround": round(
                sum(item["turnaround"] for item in metrics.values()) / len(metrics),
                4,
            )
            if metrics
            else 0,
            "average_waiting": round(
                sum(item["waiting"] for item in metrics.values()) / len(metrics),
                4,
            )
            if metrics
            else 0,
        }
        if algorithm == "RR":
            result["quantum"] = self._extract_quantum(text)
        steps = [
            f"识别调度算法为 {algorithm}，共抽取 {len(processes)} 个进程。",
            "按到达时间维护就绪队列；CPU 空闲时推进到下一个到达时刻。",
            "由时间轴计算完成时间，再计算周转时间=完成时间-到达时间，等待时间=周转时间-服务时间。",
        ]
        return self._solver_success("cpu_scheduling", result, steps)

    @staticmethod
    def _extract_vector_after_label(text: str, label: str) -> tuple[int, ...] | None:
        pattern = rf"{label}\s*(?:=|＝|:|：)\s*[（(]\s*([0-9,\s，、-]+)\s*[）)]"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        return ProblemTutoringService._parse_vector_text(match.group(1))

    @staticmethod
    def _extract_named_vectors(text: str, label: str) -> dict[str, tuple[int, ...]]:
        match = re.search(
            rf"{label}\s*(?:=|＝|:|：)\s*(.*?)(?=(?:Available|Allocation|Max|Need|Request)\s*(?:=|＝|:|：)|判断|求|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return {}
        segment = match.group(1)
        vectors: dict[str, tuple[int, ...]] = {}
        for name, vector_text in re.findall(
            r"(P\d+)\s*[（(]\s*([0-9,\s，、-]+)\s*[）)]",
            segment,
            flags=re.IGNORECASE,
        ):
            vector = ProblemTutoringService._parse_vector_text(vector_text)
            if vector is not None:
                vectors[name.upper()] = vector
        return vectors

    @staticmethod
    def _extract_banker_request(text: str) -> tuple[str | None, tuple[int, ...] | None]:
        patterns = (
            r"(P\d+)[^。；;\n]{0,40}(?:Request|请求)\s*(?:=|＝|:|：|为|是)?\s*[（(]\s*([0-9,\s，、-]+)\s*[）)]",
            r"(?:Request|请求)\s*(?:=|＝|:|：)?\s*(P\d+)\s*[（(]\s*([0-9,\s，、-]+)\s*[）)]",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                process = match.group(1).upper()
                vector = ProblemTutoringService._parse_vector_text(match.group(2))
                if vector is not None:
                    return process, vector

        vector = (
            ProblemTutoringService._extract_vector_after_label(text, "Request")
            or ProblemTutoringService._extract_vector_after_label(text, "请求")
        )
        if vector is None:
            return None, None
        process_match = re.search(
            r"(P\d+)[^。；;\n]{0,20}(?:发出|提出|申请|请求)",
            text,
            flags=re.IGNORECASE,
        )
        process = process_match.group(1).upper() if process_match else None
        return process, vector

    def _run_banker_safety(
        self,
        *,
        available: tuple[int, ...],
        allocation: dict[str, tuple[int, ...]],
        need: dict[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        process_names = sorted(set(allocation) & set(need), key=self._process_sort_key)
        work = tuple(available)
        finish = {name: False for name in process_names}
        safe_sequence: list[str] = []
        rounds: list[dict[str, Any]] = []
        while len(safe_sequence) < len(process_names):
            progressed = False
            for name in process_names:
                if finish[name] or not self._vector_leq(need[name], work):
                    continue
                before = work
                work = self._vector_add(work, allocation[name])
                finish[name] = True
                safe_sequence.append(name)
                rounds.append(
                    {
                        "process": name,
                        "need": need[name],
                        "work_before": before,
                        "allocation_released": allocation[name],
                        "work_after": work,
                    }
                )
                progressed = True
            if not progressed:
                break
        return {
            "safe": all(finish.values()),
            "safe_sequence": safe_sequence,
            "rounds": rounds,
            "finish": finish,
            "final_work": work,
        }

    def _solve_banker(self, text: str) -> dict[str, Any]:
        allocation = self._extract_named_vectors(text, "Allocation")
        maximum = self._extract_named_vectors(text, "Max")
        available = self._extract_vector_after_label(text, "Available")

        if allocation and maximum and available is not None:
            process_names = sorted(set(allocation) & set(maximum), key=self._process_sort_key)
            if not process_names:
                return self._solver_skipped("banker", "未找到同时包含 Allocation 和 Max 的进程。")
            need = {
                name: self._vector_sub(maximum[name], allocation[name])
                for name in process_names
                if len(maximum[name]) == len(allocation[name])
            }
            if len(need) != len(process_names):
                return self._solver_skipped("banker", "Allocation 和 Max 的向量维度不一致。")
            if any(len(vector) != len(available) for vector in need.values()):
                return self._solver_skipped("banker", "Available 与 Need 的资源维度不一致。")

            request_process, request = self._extract_banker_request(text)
            if request is not None:
                if request_process is None:
                    return self._solver_skipped("banker", "发现 Request 向量，但未识别请求进程。", request=request)
                if request_process not in need:
                    return self._solver_skipped(
                        "banker",
                        "请求进程不在 Allocation/Max 矩阵中。",
                        request_process=request_process,
                    )
                if len(request) != len(available):
                    return self._solver_skipped("banker", "Request 与 Available 的资源维度不一致。")

                request_leq_need = self._vector_leq(request, need[request_process])
                request_leq_available = self._vector_leq(request, available)
                if not request_leq_need or not request_leq_available:
                    result = {
                        "available": available,
                        "allocation": allocation,
                        "max": maximum,
                        "need": need,
                        "request_process": request_process,
                        "request": request,
                        "request_leq_need": request_leq_need,
                        "request_leq_available": request_leq_available,
                        "grantable": False,
                        "reject_reason": (
                            "Request 超过该进程 Need"
                            if not request_leq_need
                            else "Request 超过当前 Available"
                        ),
                    }
                    steps = [
                        "先计算 Need = Max - Allocation。",
                        f"检查 Request <= Need：{request_leq_need}。",
                        f"检查 Request <= Available：{request_leq_available}。",
                        "任一预检条件不满足时，不能进入试分配。",
                    ]
                    return self._solver_success("banker", result, steps)

                trial_available = self._vector_sub(available, request)
                trial_allocation = dict(allocation)
                trial_need = dict(need)
                trial_allocation[request_process] = self._vector_add(allocation[request_process], request)
                trial_need[request_process] = self._vector_sub(need[request_process], request)
                safety = self._run_banker_safety(
                    available=trial_available,
                    allocation=trial_allocation,
                    need=trial_need,
                )
                result = {
                    "available": available,
                    "allocation": allocation,
                    "max": maximum,
                    "need": need,
                    "request_process": request_process,
                    "request": request,
                    "request_leq_need": True,
                    "request_leq_available": True,
                    "trial_available": trial_available,
                    "trial_allocation": trial_allocation,
                    "trial_need": trial_need,
                    "safe_after_request": safety["safe"],
                    "safe_sequence": safety["safe_sequence"],
                    "rounds": safety["rounds"],
                    "grantable": bool(safety["safe"]),
                }
                steps = [
                    "先计算 Need = Max - Allocation。",
                    f"检查 {request_process} 的 Request <= Need 且 Request <= Available，预检均通过。",
                    "进行试分配：Available -= Request，Allocation += Request，Need -= Request。",
                    "对试分配后的状态执行安全性算法。",
                    "若试分配后仍存在安全序列，则该请求可以分配；否则不能分配。",
                ]
                return self._solver_success("banker", result, steps)

            safety = self._run_banker_safety(
                available=available,
                allocation=allocation,
                need=need,
            )

            result = {
                "available": available,
                "allocation": allocation,
                "max": maximum,
                "need": need,
                "safe": safety["safe"],
                "safe_sequence": safety["safe_sequence"],
                "rounds": safety["rounds"],
            }
            steps = [
                "先逐进程计算 Need = Max - Allocation。",
                f"初始 Work = Available = {available}。",
                "每轮选择 Need <= Work 的未完成进程，执行后把该进程 Allocation 释放回 Work。",
                "若所有进程都能完成，则系统处于安全状态；否则不安全。",
            ]
            return self._solver_success("banker", result, steps)

        max_vec = self._extract_vector_after_label(text, "Max")
        alloc_vec = self._extract_vector_after_label(text, "Allocation")
        if max_vec is not None and alloc_vec is not None:
            if len(max_vec) != len(alloc_vec):
                return self._solver_skipped("banker", "Max 和 Allocation 向量维度不一致。")
            need_vec = self._vector_sub(max_vec, alloc_vec)
            result = {
                "max": max_vec,
                "allocation": alloc_vec,
                "need": need_vec,
            }
            steps = [
                "Need 表示该进程还可能请求的最大资源量。",
                "逐资源类型计算 Need = Max - Allocation。",
                f"代入得到 Need = {need_vec}。",
            ]
            return self._solver_success("banker", result, steps)

        return self._solver_skipped("banker", "未抽取到完整的 Available/Allocation/Max 或 Max/Allocation 向量。")

    @staticmethod
    def _extract_buffer_capacity(text: str) -> int | None:
        return ProblemTutoringService._extract_labeled_int(
            text,
            (
                r"(?:缓冲区|缓冲池|缓冲槽|缓冲单元)(?:大小|容量|数)?\s*(?:为|是|=|:|：)?\s*(\d+)",
                r"(\d+)\s*(?:个)?(?:缓冲区|缓冲槽|缓冲单元)",
            ),
        )

    def _solve_pv_sync(self, text: str) -> dict[str, Any]:
        normalized = self._normalize_text(text)
        semaphores: list[dict[str, Any]]
        pseudocode: dict[str, list[str]]
        relation_analysis: list[str]
        scenario = "generic"

        if "生产者消费者" in normalized or ("生产者" in normalized and "消费者" in normalized):
            scenario = "producer_consumer"
            capacity = self._extract_buffer_capacity(text) or "N"
            semaphores = [
                {"name": "mutex", "initial": 1, "meaning": "互斥访问缓冲区"},
                {"name": "empty", "initial": capacity, "meaning": "空缓冲区数量"},
                {"name": "full", "initial": 0, "meaning": "满缓冲区数量"},
            ]
            pseudocode = {
                "producer": ["P(empty)", "P(mutex)", "放入产品", "V(mutex)", "V(full)"],
                "consumer": ["P(full)", "P(mutex)", "取出产品", "V(mutex)", "V(empty)"],
            }
            relation_analysis = [
                "生产者和消费者对缓冲区有互斥访问关系。",
                "生产者需要等待 empty，消费者需要等待 full，二者存在同步关系。",
            ]
        elif "读者写者" in normalized or ("读者" in normalized and "写者" in normalized):
            scenario = "reader_writer"
            semaphores = [
                {"name": "rw", "initial": 1, "meaning": "控制写者独占访问共享数据"},
                {"name": "mutex", "initial": 1, "meaning": "互斥修改 readcount"},
                {"name": "readcount", "initial": 0, "meaning": "当前读者数量，属于计数变量"},
            ]
            pseudocode = {
                "reader": [
                    "P(mutex)",
                    "readcount++；若 readcount == 1，则 P(rw)",
                    "V(mutex)",
                    "读数据",
                    "P(mutex)",
                    "readcount--；若 readcount == 0，则 V(rw)",
                    "V(mutex)",
                ],
                "writer": ["P(rw)", "写数据", "V(rw)"],
            }
            relation_analysis = [
                "多个读者可并发读，因此读者之间不互斥。",
                "写者与任何读者/写者都互斥，需要用 rw 控制共享数据访问。",
                "readcount 是共享计数变量，修改时需要 mutex 保护。",
            ]
        elif "前驱" in normalized or "先后" in normalized or "顺序" in normalized:
            scenario = "precedence"
            semaphores = [
                {"name": "s", "initial": 0, "meaning": "表示前驱事件是否已经完成"},
            ]
            pseudocode = {
                "predecessor": ["执行前驱操作", "V(s)"],
                "successor": ["P(s)", "执行后继操作"],
            }
            relation_analysis = [
                "前驱关系属于同步约束，不是共享资源互斥。",
                "同步信号量初值通常为 0，前驱完成后 V，后继开始前 P。",
            ]
        else:
            semaphores = [
                {"name": "mutex", "initial": 1, "meaning": "保护共享资源或临界区"},
                {"name": "sync", "initial": 0, "meaning": "表达先后事件约束，若题干存在同步关系再使用"},
            ]
            pseudocode = {
                "mutual_exclusion": ["P(mutex)", "访问共享资源/临界区", "V(mutex)"],
                "synchronization": ["前驱进程完成事件后 V(sync)", "后继进程执行前 P(sync)"],
            }
            relation_analysis = [
                "先找共享资源，多个进程不能同时访问时使用互斥信号量。",
                "再找必须先发生/后发生的事件，用同步信号量表达顺序约束。",
            ]

        result = {
            "scenario": scenario,
            "relation_analysis": relation_analysis,
            "semaphores": semaphores,
            "pseudocode": pseudocode,
            "caveat": "这是按典型 PV 题型生成的结构化框架；若题干给出额外约束，应在此基础上调整信号量初值和 P/V 位置。",
        }
        steps = [
            "先区分互斥关系和同步先后关系。",
            "互斥信号量保护共享临界资源，初值通常为 1。",
            "同步信号量表达事件先后，初值按初始可发生事件数设置，常见前驱约束初值为 0。",
            "把 P 操作放在等待资源/事件之前，把 V 操作放在释放资源/通知事件之后。",
        ]
        return self._solver_success("pv_sync", result, steps)

    @staticmethod
    def _looks_like_dh_problem(
        text: str,
        *,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
    ) -> bool:
        normalized = str(text or "").lower()
        if analysis.subject_id != "cybersec_lab" and template.subject_id != "cybersec_lab":
            return False
        return "dh" in normalized or "diffie" in normalized or "密钥交换" in normalized

    @staticmethod
    def _extract_dh_scalar(text: str, key: str, role_label: str | None = None) -> int | None:
        patterns = [rf"(?<![a-zA-Z]){key}(?![a-zA-Z])\s*(?:=|＝|:|：|为|是)\s*(\d+)"]
        if role_label:
            patterns.insert(
                0,
                rf"{role_label}(?:的)?私钥\s*{key}?\s*(?:=|＝|:|：|为|是)?\s*(\d+)",
            )
        return ProblemTutoringService._extract_labeled_int(text, tuple(patterns))

    def _solve_diffie_hellman(self, text: str) -> dict[str, Any]:
        p = self._extract_dh_scalar(text, "p")
        g = self._extract_dh_scalar(text, "g")
        a = self._extract_dh_scalar(text, "a", "甲")
        b = self._extract_dh_scalar(text, "b", "乙")
        if None in {p, g, a, b}:
            return self._solver_skipped(
                "diffie_hellman",
                "DH 模运算缺少 p、g、a、b 中的至少一个参数。",
                p=p,
                g=g,
                a=a,
                b=b,
            )
        if p is None or g is None or a is None or b is None:
            return self._solver_skipped("diffie_hellman", "DH 参数不完整。")
        if p <= 1:
            return self._solver_skipped("diffie_hellman", "模数 p 必须大于 1。", p=p)

        public_a = pow(g, a, p)
        public_b = pow(g, b, p)
        key_from_a = pow(public_b, a, p)
        key_from_b = pow(public_a, b, p)
        result = {
            "p": p,
            "g": g,
            "private_a": a,
            "private_b": b,
            "public_a": public_a,
            "public_b": public_b,
            "shared_key_from_a": key_from_a,
            "shared_key_from_b": key_from_b,
            "shared_key": key_from_a if key_from_a == key_from_b else None,
        }
        steps = [
            f"甲公开值 A = g^a mod p = {g}^{a} mod {p} = {public_a}。",
            f"乙公开值 B = g^b mod p = {g}^{b} mod {p} = {public_b}。",
            f"甲计算 K = B^a mod p = {public_b}^{a} mod {p} = {key_from_a}。",
            f"乙计算 K = A^b mod p = {public_a}^{b} mod {p} = {key_from_b}。",
        ]
        return self._solver_success("diffie_hellman", result, steps)

    def looks_like_tutoring_request(
        self,
        text: str,
        *,
        requested_by_user: bool = False,
    ) -> bool:
        raw = str(text or "").strip()
        normalized = self._normalize_text(raw)
        if requested_by_user:
            return bool(raw)
        if len(normalized) < 6:
            return False

        # Avoid routing product/design discussions about this module into the module itself.
        if ("模块" in normalized or "系统" in normalized) and any(
            marker in normalized for marker in ("必要", "好实现", "怎么实现", "咋实现", "设计")
        ):
            if not any(marker in normalized for marker in ("这题", "题目如下", "题干", "求解题目")):
                return False

        concept_markers = (
            "是什么",
            "什么是",
            "是啥",
            "啥是",
            "什么意思",
            "含义",
            "定义",
            "概念",
            "作用",
            "区别",
            "联系",
            "介绍一下",
            "讲讲",
            "科普",
        )
        problem_context_markers = (
            "这题",
            "这个题",
            "这道题",
            "题目如下",
            "题干",
            "解题",
            "求解",
            "分步",
            "答案怎么写",
            "考试怎么写",
            "怎么算",
            "求出",
            "计算",
            "给定",
            "已知",
            "如下",
            "访问序列",
            "页框数",
            "到达时间",
            "服务时间",
            "时间片",
            "allocation",
            "available",
            "request",
            "need",
            "max",
            "代码如下",
            "下面代码",
            "阅读代码",
            "现象分析",
            "结果分析",
            "实验步骤",
            "抓包",
            "日志",
        )
        if any(marker in normalized for marker in concept_markers) and not any(
            marker in normalized for marker in problem_context_markers
        ):
            return False

        strong_markers = (
            "clock",
            "银行家算法",
            "安全序列",
            "request",
            "进程调度",
            "srtf",
            "最短剩余时间",
            "抢占式短作业",
            "缺页次数",
            "命中率",
            "周转时间",
            "等待时间",
            "响应比",
            "时间片",
            "资源分配图",
            "pv操作",
            "p操作",
            "v操作",
            "生产者消费者",
            "读者写者",
            "输出什么",
            "运行结果",
            "程序输出",
            "找错",
            "改错",
            "哪里错",
            "段错误",
            "现象分析",
            "结果分析",
            "实验步骤",
        )
        if any(marker in normalized for marker in strong_markers):
            return True

        has_code = "```" in raw or ("#include" in raw and "main" in raw)
        if has_code and any(marker in normalized for marker in ("输出", "运行", "分析", "错误", "这段代码")):
            return True

        algorithm_markers = (
            "页面置换",
            "缺页",
            "银行家算法",
            "进程调度",
            "死锁",
            "信号量",
            "生产者消费者",
            "读者写者",
            "实验现象",
            "结果分析",
            "实验步骤",
        )
        target_markers = (
            "求",
            "计算",
            "统计",
            "判断",
            "分析",
            "写出",
            "给出",
            "输出",
            "修改",
            "排查",
            "验证",
        )
        detail_markers = (
            "访问序列",
            "页框数",
            "到达时间",
            "服务时间",
            "时间片",
            "allocation",
            "available",
            "request",
            "need",
            "max",
            "资源分配图",
            "缓冲区",
            "私钥",
            "公钥",
            "共享密钥",
            "抓包",
            "日志",
            "现象",
        )
        has_algorithm_marker = any(marker in normalized for marker in algorithm_markers)
        has_target_marker = any(marker in normalized for marker in target_markers)
        has_detail_marker = any(marker in normalized for marker in detail_markers)
        if has_algorithm_marker and (has_detail_marker or has_target_marker) and (
            has_detail_marker or bool(re.search(r"\d", raw))
        ):
            return True

        exercise_markers = (
            "这题",
            "这个题",
            "这道题",
            "题目如下",
            "题干",
            "解题",
            "求解",
            "分步",
            "答案怎么写",
            "考试怎么写",
            "怎么算",
            "求出",
            "计算",
        )
        return any(marker in normalized for marker in exercise_markers)

    def match_request(
        self,
        text: str,
        *,
        requested_subjects: list[str] | None = None,
        requested_by_user: bool = False,
    ) -> dict[str, Any] | None:
        if not self.looks_like_tutoring_request(
            text,
            requested_by_user=requested_by_user,
        ):
            return None
        subject_hint = requested_subjects[0] if requested_subjects else ""
        analysis = self.rule_analyze(text, subject_id_hint=subject_hint or None)
        return {
            "trigger": "explicit" if requested_by_user else "auto",
            "analysis": self._model_to_dict(analysis),
        }

    def rule_analyze(
        self,
        text: str,
        *,
        subject_id_hint: str | None = None,
    ) -> TutoringProblemAnalysis:
        subject_id = self._pick_subject_by_rules(text, subject_id_hint=subject_id_hint)
        problem_type, answer_focus = self._classify_problem_type(text, subject_id)
        knowledge_points = self._extract_knowledge_points(text, subject_id, problem_type)
        target = self._extract_target(text)
        conditions = self._extract_conditions(text)
        confidence = 0.82 if problem_type != "general_problem" else 0.60
        return TutoringProblemAnalysis(
            is_problem=True,
            subject_id=subject_id,  # type: ignore[arg-type]
            problem_type=problem_type,
            confidence=confidence,
            target=target,
            extracted_conditions=conditions,
            knowledge_points=knowledge_points,
            answer_focus=answer_focus,  # type: ignore[arg-type]
            reason="规则识别题目辅导请求",
        )

    def _pick_subject_by_rules(
        self,
        text: str,
        *,
        subject_id_hint: str | None = None,
    ) -> str:
        if subject_id_hint in {"C_program", "operating_systems", "cybersec_lab"}:
            return subject_id_hint

        normalized = self._normalize_text(text)
        scores = {
            "C_program": 0,
            "operating_systems": 0,
            "cybersec_lab": 0,
        }
        for marker in (
            "#include",
            "printf",
            "scanf",
            "malloc",
            "free",
            "指针",
            "数组",
            "结构体",
            "函数",
            "阅读代码",
            "程序输出",
            "找错",
            "改错",
            "int*",
            "return&",
        ):
            if marker.lower() in normalized:
                scores["C_program"] += 1
        for marker in (
            "页面置换",
            "缺页",
            "clock",
            "进程调度",
            "调度",
            "fcfs",
            "sjf",
            "srtf",
            "最短剩余时间",
            "抢占式短作业",
            "rr",
            "时间片",
            "完成时间",
            "服务时间",
            "到达时间",
            "银行家",
            "死锁",
            "信号量",
            "pv",
            "页框",
            "周转时间",
            "等待时间",
            "操作系统",
        ):
            if marker in normalized:
                scores["operating_systems"] += 1
        for marker in (
            "网络安全",
            "安全实验",
            "安全通信",
            "加密",
            "密钥",
            "会话密钥",
            "认证",
            "访问控制",
            "访问与职责",
            "角色",
            "权限",
            "最小权限",
            "数字签名",
            "安全审计",
            "审计日志",
            "抓包",
            "dh",
            "des",
            "实验现象",
        ):
            if marker in normalized:
                scores["cybersec_lab"] += 1

        subject_id, score = max(scores.items(), key=lambda item: item[1])
        return subject_id if score > 0 else "unknown"

    def _classify_problem_type(self, text: str, subject_id: str) -> tuple[str, str]:
        normalized = self._normalize_text(text)
        if subject_id == "C_program":
            if any(marker in normalized for marker in ("找错", "改错", "报错", "哪里错", "段错误", "debug")):
                return "c_debug", "debugging"
            function_defs = re.findall(
                r"\b(?:void|int|char|float|double|long|short)\s+\w+\s*\([^;{}]*\)\s*\{",
                text,
            )
            if (
                any(marker in normalized for marker in ("函数调用", "形参", "实参", "返回值", "递归"))
                or len(function_defs) >= 2
            ):
                return "c_function_call", "code_reading"
            if any(
                marker in normalized
                for marker in (
                    "指针",
                    "数组",
                    "解引用",
                    "地址",
                    "下标",
                    "int*p",
                    "char*",
                    "*(p",
                    "p++",
                    "p+=",
                    "&a[",
                )
            ):
                return "c_pointer_array", "code_reading"
            if any(marker in normalized for marker in ("输出什么", "运行结果", "程序输出", "printf")):
                return "c_output", "code_reading"
            return "general_problem", "general"

        if subject_id == "operating_systems":
            if any(
                marker in normalized
                for marker in ("页面置换", "缺页次数", "缺页率", "命中率", "页框", "fifo", "lru", "opt", "clock")
            ):
                return "os_page_replacement", "calculation"
            if any(
                marker in normalized
                for marker in (
                    "进程调度",
                    "fcfs",
                    "sjf",
                    "srtf",
                    "最短剩余时间",
                    "抢占式短作业",
                    "rr",
                    "时间片",
                    "周转时间",
                    "等待时间",
                    "响应比",
                )
            ):
                return "os_cpu_scheduling", "calculation"
            if any(marker in normalized for marker in ("银行家", "安全序列", "available", "allocation", "request", "need", "max")):
                return "os_banker", "calculation"
            if any(marker in normalized for marker in ("死锁", "资源分配图", "循环等待", "安全状态")):
                return "os_deadlock", "conceptual"
            if any(marker in normalized for marker in ("pv", "p操作", "v操作", "信号量", "同步", "互斥", "生产者消费者", "读者写者")):
                return "os_pv_sync", "conceptual"
            return "general_problem", "general"

        if subject_id == "cybersec_lab":
            if any(marker in normalized for marker in ("实验步骤", "怎么做", "如何做", "配置", "搭建", "结果验证")):
                return "cybersec_lab_steps", "lab_analysis"
            if any(
                marker in normalized
                for marker in (
                    "现象分析",
                    "结果分析",
                    "抓包",
                    "日志",
                    "失败原因",
                    "解密失败",
                    "中间人",
                    "dh",
                    "diffie",
                )
            ):
                return "cybersec_phenomenon_analysis", "lab_analysis"
            return "general_problem", "lab_analysis"

        if any(
            marker in normalized
            for marker in (
                "页面置换",
                "进程调度",
                "银行家",
                "死锁",
                "信号量",
                "fcfs",
                "sjf",
                "srtf",
                "最短剩余时间",
                "抢占式短作业",
                "rr",
                "时间片",
                "周转时间",
                "等待时间",
                "完成时间",
                "到达时间",
            )
        ):
            return self._classify_problem_type(text, "operating_systems")
        if any(marker in normalized for marker in ("printf", "#include", "指针", "数组")):
            return self._classify_problem_type(text, "C_program")
        if any(marker in normalized for marker in ("实验", "加密", "认证", "抓包")):
            return self._classify_problem_type(text, "cybersec_lab")
        return "general_problem", "general"

    def _extract_knowledge_points(
        self,
        text: str,
        subject_id: str,
        problem_type: str,
    ) -> list[str]:
        normalized = self._normalize_text(text)
        points: list[str] = []
        templates = [
            item
            for item in self.TEMPLATES
            if item.subject_id == subject_id and item.problem_type == problem_type
        ]
        if templates:
            points.extend(templates[0].knowledge_keywords[:4])

        extra_markers = {
            "FIFO": ("fifo", "FIFO"),
            "LRU": ("lru", "LRU"),
            "OPT": ("opt", "OPT"),
            "时间片轮转": ("时间片", "RR"),
            "银行家算法": ("银行家", "Need", "Available"),
            "指针": ("指针", "解引用", "地址"),
            "数组": ("数组", "下标", "越界"),
            "Diffie-Hellman": ("dh", "diffie"),
            "DES": ("des",),
        }
        for label, markers in extra_markers.items():
            if any(marker.lower() in normalized for marker in markers):
                points.append(label)

        seen: set[str] = set()
        deduped = []
        for point in points:
            key = str(point).strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped[:8]

    def _extract_target(self, text: str) -> str:
        raw = str(text or "").strip()
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        question_lines = [
            line for line in lines if any(marker in line for marker in ("?", "？", "求", "计算", "输出", "分析", "判断"))
        ]
        if question_lines:
            return self._short_text(question_lines[-1], max_len=240)
        return self._short_text(lines[0] if lines else raw, max_len=240)

    def _extract_conditions(self, text: str) -> list[str]:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        selected: list[str] = []
        for line in lines:
            if len(selected) >= 10:
                break
            if "```" in line:
                continue
            if len(line) <= 180 and (
                re.search(r"\d", line)
                or any(marker in line for marker in ("进程", "页", "框", "Available", "Allocation", "Max", "Need", "printf", "scanf", "实验", "现象"))
            ):
                selected.append(line)
        if not selected and lines:
            selected = lines[: min(5, len(lines))]
        return selected

    async def analyze_question(
        self,
        *,
        user_question: str,
        subject_id_hint: str | None,
        timeout_s: int,
    ) -> TutoringProblemAnalysis:
        rule_obj = self.rule_analyze(user_question, subject_id_hint=subject_id_hint)
        if self.llm_analysis_struct is None:
            return rule_obj

        prompt = (
            "你是课程题目辅导模块的题目解析器。请判断用户输入是否是课程题目辅导请求，"
            "并抽取学科、题型、求解目标、关键条件和知识点。\n"
            "学科ID只能从 C_program / operating_systems / cybersec_lab / unknown 中选择。\n"
            "优先题型ID：\n"
            "- C语言：c_output, c_debug, c_pointer_array, c_function_call, general_problem\n"
            "- 操作系统：os_page_replacement, os_cpu_scheduling, os_banker, os_deadlock, os_pv_sync, general_problem\n"
            "- 网络安全实验：cybersec_lab_steps, cybersec_phenomenon_analysis, general_problem\n"
            "如果题型不确定，使用 general_problem，但不要编造题干没有的条件。\n\n"
            f"学科提示：{subject_id_hint or '无'}\n"
            f"用户输入：\n{self._short_text(user_question, max_len=6000)}"
        )
        try:
            obj = await asyncio.wait_for(
                self.llm_analysis_struct.ainvoke(prompt),
                timeout=max(1, min(int(timeout_s), 6)),
            )
            data = self._model_to_dict(obj)
            if subject_id_hint in {"C_program", "operating_systems", "cybersec_lab"}:
                data["subject_id"] = subject_id_hint
            if not data.get("problem_type") or data.get("problem_type") == "unknown":
                data["problem_type"] = rule_obj.problem_type
            if not data.get("knowledge_points"):
                data["knowledge_points"] = rule_obj.knowledge_points
            if not data.get("extracted_conditions"):
                data["extracted_conditions"] = rule_obj.extracted_conditions
            if not data.get("target"):
                data["target"] = rule_obj.target
            if not data.get("answer_focus"):
                data["answer_focus"] = rule_obj.answer_focus
            data["is_problem"] = bool(data.get("is_problem", True))
            return TutoringProblemAnalysis(**data)
        except Exception:
            return rule_obj

    def select_template(
        self,
        analysis: TutoringProblemAnalysis,
        *,
        user_question: str = "",
    ) -> ProblemTemplate:
        subject_id = analysis.subject_id if analysis.subject_id != "unknown" else self._pick_subject_by_rules(user_question)
        problem_type = str(analysis.problem_type or "general_problem").strip()

        exact = [
            item
            for item in self.TEMPLATES
            if item.subject_id == subject_id and item.problem_type == problem_type
        ]
        if exact:
            return sorted(exact, key=lambda item: item.priority, reverse=True)[0]

        normalized = self._normalize_text(user_question)
        candidates = [item for item in self.TEMPLATES if item.subject_id == subject_id]
        scored: list[tuple[int, ProblemTemplate]] = []
        for item in candidates:
            score = item.priority
            for alias in item.aliases:
                if self._normalize_text(alias) in normalized:
                    score += 30
            for keyword in item.knowledge_keywords:
                if self._normalize_text(keyword) in normalized:
                    score += 10
            scored.append((score, item))
        if scored:
            return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]

        fallback_id = {
            "operating_systems": "os_general_problem",
            "cybersec_lab": "cybersec_general_problem",
        }.get(subject_id, "subject_general_problem")
        return next(item for item in self.TEMPLATES if item.id == fallback_id)

    async def retrieve_knowledge(
        self,
        *,
        user_question: str,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        working_dir: str | None,
        timeout_s: int,
    ) -> dict[str, Any]:
        if not working_dir:
            return {
                "status": "skipped",
                "message": "缺少学科知识库 working_dir",
                "answer": "",
            }

        subject_label = self.SUBJECT_LABELS.get(analysis.subject_id, analysis.subject_id)
        query = (
            f"课程：{subject_label}\n"
            f"题型：{template.name}\n"
            f"知识点：{', '.join(analysis.knowledge_points or template.knowledge_keywords)}\n"
            "请检索与这道题解题步骤、算法规则、概念依据有关的课程内容。"
            "不要直接发挥解答，优先返回规则、公式、定义、步骤依据。\n"
            f"题目：{self._short_text(user_question, max_len=3000)}"
        )
        try:
            from lightrag import QueryParam

            from agenticRAG.agentic_runtime import get_rag
            from agenticRAG.query_utils import extract_query_response_fields

            rag = await get_rag(working_dir)
            query_resp = await asyncio.wait_for(
                rag.aquery_llm(
                    query,
                    param=QueryParam(
                        mode="hybrid",
                        top_k=10,
                        chunk_top_k=8,
                        enable_rerank=False,
                        response_type=(
                            "请用中文提取本题相关的知识点、算法规则、计算公式、实验依据；"
                            "若材料不足，请明确指出不足。"
                        ),
                    ),
                ),
                timeout=max(1, min(int(timeout_s), 12)),
            )
            status, message, failure_reason, answer = extract_query_response_fields(query_resp)
            return {
                "status": status,
                "message": message,
                "failure_reason": failure_reason,
                "answer": str(answer or "").strip(),
            }
        except Exception as exc:
            return {
                "status": "failure",
                "message": self._normalize_exception_message(exc),
                "failure_reason": type(exc).__name__,
                "answer": "",
            }

    async def prepare(
        self,
        *,
        user_question: str,
        augmented_question: str,
        subject_id_hint: str | None,
        working_dir: str | None,
        mode: str,
        response_language: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        analysis_budget = max(2, min(int(timeout_s), 6))
        analysis = await self.analyze_question(
            user_question=user_question,
            subject_id_hint=subject_id_hint,
            timeout_s=analysis_budget,
        )
        if subject_id_hint in {"C_program", "operating_systems", "cybersec_lab"} and analysis.subject_id != subject_id_hint:
            analysis = TutoringProblemAnalysis(
                **{
                    **self._model_to_dict(analysis),
                    "subject_id": subject_id_hint,
                }
        )
        template = self.select_template(analysis, user_question=user_question)
        recommendations = await self.recommend_similar_questions_hybrid(
            analysis=analysis,
            template=template,
            user_question=augmented_question,
            limit=3,
        )
        few_shot_examples = self.build_few_shot_examples(recommendations, limit=2)
        solver_result = self.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=augmented_question,
        )

        elapsed = int(time.perf_counter() - started)
        retrieval_budget = max(1, min(12, int(timeout_s) - elapsed))
        retrieval = await self.retrieve_knowledge(
            user_question=augmented_question,
            analysis=analysis,
            template=template,
            working_dir=working_dir,
            timeout_s=retrieval_budget,
        )
        learning_outline = self.build_learning_outline(
            analysis=analysis,
            template=template,
            recommendations=recommendations,
            few_shot_examples=few_shot_examples,
            solver_result=solver_result,
            retrieval=retrieval,
        )
        prompt = self.build_final_prompt(
            user_question=user_question,
            augmented_question=augmented_question,
            analysis=analysis,
            template=template,
            retrieval=retrieval,
            mode=mode,
            response_language=response_language,
            recommendations=recommendations,
            few_shot_examples=few_shot_examples,
            solver_result=solver_result,
        )
        return {
            "prompt": prompt,
            "analysis": self._model_to_dict(analysis),
            "template": template.to_public_dict(),
            "retrieval": retrieval,
            "question_bank_recommendations": recommendations,
            "question_bank_few_shot_examples": few_shot_examples,
            "solver_result": solver_result,
            "learning_outline": learning_outline,
            "elapsed_ms": str(int((time.perf_counter() - started) * 1000)),
        }

    def build_final_prompt(
        self,
        *,
        user_question: str,
        augmented_question: str,
        analysis: TutoringProblemAnalysis,
        template: ProblemTemplate,
        retrieval: dict[str, Any],
        mode: str,
        response_language: str,
        recommendations: list[dict[str, Any]] | None = None,
        few_shot_examples: list[dict[str, Any]] | None = None,
        solver_result: dict[str, Any] | None = None,
    ) -> str:
        language_instruction = (
            "Answer entirely in English. Do not use Chinese characters."
            if response_language == "en"
            else "请全程使用中文回答，必要术语可保留英文缩写。"
        )
        analysis_json = json.dumps(
            self._model_to_dict(analysis),
            ensure_ascii=False,
            indent=2,
        )
        retrieval_text = self._short_text(str(retrieval.get("answer", "") or ""), max_len=5000)
        retrieval_status = json.dumps(
            {
                "status": retrieval.get("status", ""),
                "message": retrieval.get("message", ""),
                "failure_reason": retrieval.get("failure_reason", ""),
            },
            ensure_ascii=False,
        )
        recommendation_text = json.dumps(
            list(recommendations or []),
            ensure_ascii=False,
            indent=2,
        )
        few_shot_text = json.dumps(
            list(few_shot_examples or []),
            ensure_ascii=False,
            indent=2,
        )
        solver_text = json.dumps(
            solver_result or {
                "status": "skipped",
                "message": "未运行确定性规则求解器。",
            },
            ensure_ascii=False,
            indent=2,
        )
        return (
            "你是面向 C语言、操作系统、网络安全实验的课程题目辅导助手。"
            "你的任务不是只给最终答案，而是展示可追踪的解题过程。\n\n"
            f"回答语言要求：{language_instruction}\n"
            f"当前模式：{mode}\n\n"
            "必须遵守：\n"
            "1) 先判断题型和考点，再分步求解；\n"
            "2) 若题干条件不足，明确列出缺失条件，不要硬算；\n"
            "3) 若课程检索依据不足，可以基于通用课程知识推理，但必须标注“依据不足/按常规定义”；\n"
            "4) C 程序输出题不要假装实际运行，必须按语句手动跟踪；遇到未定义行为要先指出输出不稳定；\n"
            "5) 操作系统计算题优先给过程表、甘特图或状态变化；\n"
            "6) 网络安全实验题只面向课程实验和授权环境，不给真实未授权攻击指导；\n"
            "7) 类题推荐阶段优先使用下方“题库相似题推荐”中的真实题号和题干；"
            "如果列表为空，再给训练方向；不要编造未提供的题号；\n"
            "8) 若“确定性规则求解结果”的 status 为 success，必须优先采用其中的数值、过程和结论；"
            "不要给出与该结果矛盾的计算答案；\n"
            "9) “相似题标准解法示例”只用于模仿解题结构、步骤粒度和易错点表达；"
            "当前题答案必须依据用户原题，不得把示例题的具体数值、题号或结论当作当前题答案。\n\n"
            "建议输出结构：\n"
            "## 题型判断\n"
            "## 考点定位\n"
            "## 条件抽取\n"
            "## 解题过程\n"
            "## 结论 / 考试写法\n"
            "## 易错点\n"
            "## 类题训练方向\n\n"
            "[结构化题目解析]\n"
            f"```json\n{analysis_json}\n```\n\n"
            "[匹配到的解题模板]\n"
            f"```json\n{template.to_prompt_block()}\n```\n\n"
            "[确定性规则求解结果]\n"
            f"```json\n{solver_text}\n```\n\n"
            "[课程检索状态]\n"
            f"```json\n{retrieval_status}\n```\n\n"
            "[课程检索依据]\n"
            f"{retrieval_text or '未检索到可用依据。'}\n\n"
            "[题库相似题推荐]\n"
            f"```json\n{recommendation_text}\n```\n\n"
            "[相似题标准解法示例]\n"
            f"```json\n{few_shot_text}\n```\n\n"
            "[用户原题]\n"
            f"{self._short_text(user_question, max_len=6000)}\n\n"
            "[多轮上下文增强题目]\n"
            f"{self._short_text(augmented_question, max_len=6000)}"
        )
