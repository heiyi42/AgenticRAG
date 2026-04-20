from __future__ import annotations

import re
from threading import Lock
from typing import Any, Callable

import models.auto as auto

from . import config as cfg
from .chat_auto_orchestration import ChatAutoOrchestrationMixin
from .chat_code_analysis import ChatCodeAnalysisMixin
from .chat_retrieval_support import ChatRetrievalSupportMixin
from .chat_streaming import ChatStreamingMixin
from .chat_routing import (
    ChatRoutingMixin,
    RetrievalGateDecision,
    SubjectRouteDecision,
)
from .code_analysis_service import CodeAnalysisService
from .problem_tutoring_service import ProblemTutoringService
from .session_store import SessionStore



class ChatService(
    ChatStreamingMixin,
    ChatAutoOrchestrationMixin,
    ChatRetrievalSupportMixin,
    ChatCodeAnalysisMixin,
    ChatRoutingMixin,
):
    GATE_PROMPT_VERSION = "v7-lightweight-retrieval-gate"
    SUBJECT_LABELS = {
        "C_program": "C语言",
        "operating_systems": "操作系统",
        "cybersec_lab": "网络安全实验",
    }
    SUBJECT_SCORE_FIELDS = {
        "C_program": "c_program_score",
        "operating_systems": "operating_systems_score",
        "cybersec_lab": "cybersec_lab_score",
    }
    SUBJECT_KEYWORDS = {
        "C_program": [
            "c语言",
            "c 程序",
            "指针",
            "数组",
            "结构体",
            "联合体",
            "宏",
            "预处理",
            "malloc",
            "free",
            "scanf",
            "printf",
            "文件io",
        ],
        "operating_systems": [
            "操作系统",
            "进程",
            "线程",
            "调度",
            "并发",
            "同步",
            "互斥",
            "死锁",
            "内存",
            "虚拟内存",
            "分页",
            "文件系统",
            "磁盘",
            "中断",
            "系统调用",
            "栈帧",
        ],
        "cybersec_lab": [
            "网络安全",
            "安全实验",
            "加密",
            "认证",
            "访问控制",
            "数字签名",
            "安全审计",
            "漏洞",
            "溢出",
            "攻击",
            "利用",
            "防护",
            "wireshark",
            "协议",
            "在线考试系统",
        ],
    }

    def __init__(
        self,
        store: SessionStore,
        run_async: Callable[[Any], Any],
        submit_async: Callable[[Any], Any] | None = None,
    ) -> None:
        self.store = store
        self.run_async = run_async
        self.submit_async = submit_async
        self.llm_retrieval_gate_struct = auto.auto_router_llm.with_structured_output(
            RetrievalGateDecision
        )
        self.llm_subject_router_struct = auto.auto_router_llm.with_structured_output(
            SubjectRouteDecision
        )
        self.code_analysis_service = CodeAnalysisService(
            compiler_bin=cfg.WEB_CODE_ANALYSIS_COMPILER or None,
            max_code_chars=cfg.WEB_CODE_ANALYSIS_MAX_CHARS,
            enable_execution=cfg.WEB_ENABLE_CODE_EXECUTION,
            execution_timeout_s=cfg.WEB_CODE_EXECUTION_TIMEOUT_S,
            execution_max_code_chars=cfg.WEB_CODE_EXECUTION_MAX_CHARS,
            execution_max_output_chars=cfg.WEB_CODE_EXECUTION_MAX_OUTPUT_CHARS,
        )
        self.problem_tutoring_service = ProblemTutoringService(auto.auto_router_llm)
        self.subject_catalog = self._build_subject_catalog()
        self._retrieval_gate_cache: dict[str, tuple[bool, float, str, float]] = {}
        self._retrieval_gate_cache_lock = Lock()
        self._event_subscribers: set[Any] = set()
        self._event_subscribers_lock = Lock()

    @staticmethod
    def _response_language_from_requested_subjects(
        requested_subjects: list[str] | None,
    ) -> str:
        subjects = list(requested_subjects or [])
        if subjects == ["operating_systems"]:
            return "en"
        return "zh"

    @staticmethod
    def _response_language_instruction(response_language: str) -> str:
        if response_language == "en":
            return (
                "Answer entirely in English. Do not use Chinese characters. "
                "Keep technical explanations in natural academic English."
            )
        return "请全程使用中文回答，必要术语可保留英文缩写，但主体说明必须是中文。"

    def _apply_response_language_to_question(
        self,
        question: str,
        response_language: str,
    ) -> str:
        instruction = self._response_language_instruction(response_language)
        text = str(question or "").rstrip()
        if not text:
            return instruction
        return f"{text}\n\n[Answer language requirement]\n{instruction}"

    @staticmethod
    def _fast_subject_route_meta(reason: str) -> dict[str, Any]:
        return {
            "primary_subject": "",
            "cross_subject": False,
            "confidence": 1.0,
            "reason": str(reason or "本地快路径"),
            "requested_subjects": [],
            "ranked": [],
        }

    def _match_code_analysis_request(
        self,
        text: str,
        *,
        requested_subjects: list[str] | None = None,
        requested_by_user: bool = False,
    ) -> dict[str, Any] | None:
        if not cfg.WEB_ENABLE_CODE_ANALYSIS:
            return None
        explicit_subjects = list(requested_subjects or [])
        if explicit_subjects and explicit_subjects != ["C_program"]:
            return None
        candidate = self.code_analysis_service.extract_c_code_candidate(text)
        if candidate is None:
            return None

        if requested_by_user:
            candidate["trigger"] = "explicit"
            return candidate

        task_type = self._detect_subject_task_type("C_program", text)
        if task_type not in {"debug", "code_reading"}:
            return None

        candidate["trigger"] = "auto"
        candidate["task_type"] = task_type
        return candidate

    def _build_code_analysis_subject_route(
        self,
        *,
        requested_subjects: list[str] | None = None,
    ) -> dict[str, Any]:
        scores = {subject_id: 0.0 for subject_id in self.SUBJECT_LABELS}
        scores["C_program"] = 1.0
        return self._subject_route_from_scores(
            scores,
            confidence=1.0,
            reason="命中 C 代码分析快路径",
            requested_subjects=requested_subjects,
        )

    @staticmethod
    def _normalize_smalltalk_text(text: str) -> str:
        return re.sub(r"[\s\u3000，。！？!?,.;；:：、~～]+", "", str(text or "")).lower()

    @staticmethod
    def _safe_bool(raw: Any, default: bool = False) -> bool:
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return default
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return default

    @classmethod
    def _fast_casual_observation_answer(
        cls,
        normalized: str,
        *,
        response_language: str = "zh",
    ) -> str | None:
        if not normalized or len(normalized) > 20:
            return None

        # Keep course-related prompts out of local闲聊快路径.
        for keywords in cls.SUBJECT_KEYWORDS.values():
            for keyword in keywords:
                marker = str(keyword or "").strip().lower()
                if marker and marker in normalized:
                    return None

        task_markers = (
            "什么",
            "怎么",
            "如何",
            "为什么",
            "帮我",
            "请",
            "介绍",
            "解释",
            "分析",
            "总结",
            "推荐",
            "比较",
            "区别",
            "实现",
            "步骤",
            "原理",
            "代码",
            "是否",
            "吗",
            "嘛",
            "呢",
            "?",
            "？",
        )
        if any(marker in normalized for marker in task_markers):
            return None

        weather_markers = (
            "天气",
            "下雨",
            "雨天",
            "晴天",
            "阴天",
            "多云",
            "刮风",
            "出太阳",
            "太阳真好",
            "好热",
            "好冷",
            "有点热",
            "有点冷",
        )
        mood_markers = (
            "开心",
            "高兴",
            "心情不错",
            "心情很好",
            "有点累",
            "好累",
            "有点困",
            "好困",
            "有点忙",
            "真忙",
            "有点烦",
            "有点郁闷",
        )
        positive_markers = ("不错", "挺好", "真好", "很好", "舒服", "美好")

        if any(marker in normalized for marker in weather_markers):
            if any(marker in normalized for marker in ("冷", "降温")):
                return (
                    "Sounds chilly today. Stay warm."
                    if response_language == "en"
                    else "是有点冷，注意保暖。你今天要出门吗？"
                )
            if any(marker in normalized for marker in ("热", "升温")):
                return (
                    "Sounds pretty warm today. Stay hydrated."
                    if response_language == "en"
                    else "是有点热，记得补水。你今天准备怎么安排？"
                )
            return (
                "The weather does sound nice. Feels like a good day to get outside."
                if response_language == "en"
                else "是啊，这种天气挺适合出去走走。你今天有什么安排？"
            )

        if any(marker in normalized for marker in mood_markers):
            if any(marker in normalized for marker in ("累", "困", "忙", "烦", "郁闷")):
                return (
                    "Sounds like you've had a lot on your plate. Take it easy."
                    if response_language == "en"
                    else "听起来你今天有点累。先缓一缓，有需要我也可以帮你分担点事情。"
                )
            return (
                "Glad to hear that. Hope the rest of your day goes well."
                if response_language == "en"
                else "那挺好，希望你今天都能保持这个状态。"
            )

        if any(marker in normalized for marker in positive_markers):
            return (
                "Sounds good."
                if response_language == "en"
                else "确实，听起来不错。"
            )

        return None

    def _fast_smalltalk_answer(
        self,
        text: str,
        *,
        response_language: str = "zh",
    ) -> str | None:
        normalized = self._normalize_smalltalk_text(text)
        if not normalized or len(normalized) > 18:
            return None

        greeting_pattern = re.compile(
            r"(你好(呀|啊)?|您好(呀|啊)?|嗨(喽)?(呀|啊)?|哈喽|hello|hi|hey|"
            r"早上好|上午好|中午好|下午好|晚上好|在吗|在嘛|在不在)"
        )
        status_pattern = re.compile(
            r"(你好吗|你还好吗|最近怎么样|最近还好吗|最近咋样|过得怎么样)"
        )
        thanks_pattern = re.compile(
            r"(谢谢(你)?(啦)?|多谢|感谢|thanks|thankyou|thanku)"
        )
        identity_pattern = re.compile(
            r"(你是?谁(啊|呀)?|你叫什么(名字)?|你叫啥|你是(干嘛|做什么)的|"
            r"你能做什么(啊|呀)?|你会什么(啊|呀)?|你有什么功能(啊|呀)?|"
            r"介绍一下你自己|自我介绍一下)"
        )
        farewell_pattern = re.compile(r"(拜拜|白白|再见|回头见|bye|goodbye)")
        laughter = (
            normalized in {"呵呵", "嘿嘿", "hh", "hhh", "lol"}
            or (len(normalized) >= 2 and set(normalized) <= {"哈"})
            or (len(normalized) >= 2 and set(normalized) <= {"呵"})
            or (len(normalized) >= 2 and set(normalized) <= {"嘿"})
        )

        if greeting_pattern.fullmatch(normalized):
            return (
                "Hello. How can I help?"
                if response_language == "en"
                else "你好！有什么我可以帮你的？"
            )
        if status_pattern.fullmatch(normalized):
            return (
                "I'm doing well, thanks. How about you?"
                if response_language == "en"
                else "我很好，谢谢。你呢？"
            )
        if thanks_pattern.fullmatch(normalized):
            return "You're welcome." if response_language == "en" else "不客气。"
        if identity_pattern.fullmatch(normalized):
            return (
                "I'm the assistant in this app. I can chat with you and also help answer "
                "questions about C programming, operating systems, and cybersecurity labs."
                if response_language == "en"
                else "我是这个应用里的智能助理，可以陪你聊天，也可以帮你回答 C语言、操作系统、网络安全实验 这几个方向的问题。"
            )
        if laughter:
            return (
                "Sounds like you're in a good mood. Want to chat, or do you want help with something?"
                if response_language == "en"
                else "看来你心情不错。想随便聊聊，还是要我帮你处理点什么？"
            )
        if farewell_pattern.fullmatch(normalized):
            return "Bye." if response_language == "en" else "回头见。"
        return self._fast_casual_observation_answer(
            normalized,
            response_language=response_language,
        )

    def _fast_smalltalk_result_bundle(
        self,
        *,
        mode: str,
        text: str,
        response_language: str = "zh",
    ) -> tuple[dict[str, Any], bool, float, str] | None:
        answer = self._fast_smalltalk_answer(
            text,
            response_language=response_language,
        )
        if not answer:
            return None
        return (
            {
                "mode_used": mode,
                "answer": answer,
                "route": {"chain": "direct-local", "reason": "smalltalk_fast_path"},
                "subject_route": self._fast_subject_route_meta("寒暄本地快路径"),
                "upgraded": False,
                "upgrade_reason": "",
                "instant_review": {
                    "heuristic": "",
                    "review": "direct_smalltalk_fast",
                },
            },
            False,
            1.0,
            "本地寒暄快路径",
        )

    @staticmethod
    def _normalize_task_type_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).lower()

    @classmethod
    def _detect_subject_task_type(cls, subject_id: str, text: str) -> str:
        normalized = cls._normalize_task_type_text(text)
        if not normalized:
            return "general"

        if subject_id == "C_program":
            if any(
                marker in normalized
                for marker in (
                    "报错",
                    "错误",
                    "warning",
                    "debug",
                    "调试",
                    "段错误",
                    "segmentationfault",
                    "哪里错",
                    "为什么错",
                    "无法运行",
                    "崩溃",
                )
            ):
                return "debug"
            if any(
                marker in normalized
                for marker in (
                    "这段代码",
                    "下面代码",
                    "代码如下",
                    "阅读代码",
                    "分析代码",
                    "看代码",
                    "输出什么",
                    "运行结果",
                )
            ) or "```" in str(text or ""):
                return "code_reading"
            if any(
                marker in normalized
                for marker in (
                    "是什么",
                    "什么是",
                    "定义",
                    "作用",
                    "区别",
                    "原理",
                    "含义",
                    "概念",
                    "什么意思",
                    "如何理解",
                )
            ):
                return "concept"
            return "general"

        if subject_id == "operating_systems":
            if any(
                marker in normalized
                for marker in (
                    "区别",
                    "比较",
                    "对比",
                    "联系与区别",
                    "优缺点",
                    "异同",
                    "哪个好",
                )
            ):
                return "comparison"
            if any(
                marker in normalized
                for marker in (
                    "机制",
                    "过程",
                    "流程",
                    "如何",
                    "怎么",
                    "原理",
                    "为什么",
                    "是怎样",
                    "工作方式",
                )
            ):
                return "mechanism"
            if any(
                marker in normalized
                for marker in (
                    "是什么",
                    "什么是",
                    "定义",
                    "含义",
                    "概念",
                )
            ):
                return "concept"
            return "general"

        if subject_id == "cybersec_lab":
            if any(
                marker in normalized
                for marker in (
                    "报错",
                    "错误",
                    "失败",
                    "异常",
                    "排查",
                    "无法",
                    "不行",
                    "连不上",
                    "没反应",
                    "问题出在哪",
                )
            ):
                return "troubleshooting"
            if any(
                marker in normalized
                for marker in (
                    "步骤",
                    "流程",
                    "怎么做",
                    "如何做",
                    "实验",
                    "配置",
                    "搭建",
                    "命令",
                    "使用",
                )
            ):
                return "lab_steps"
            return "general"

        return "general"

    @staticmethod
    def _subject_answer_style_instruction(
        subject_id: str,
        task_type: str,
        mode: str,
        response_language: str,
    ) -> str:
        mode_id = str(mode or "auto").strip().lower()
        if subject_id == "C_program":
            if task_type == "concept":
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 结论`：先用一句话直接回答问题；\n"
                    "2) `## 核心概念`：解释涉及的语法、语义或机制；\n"
                    "3) `## 代码示例`：给出最小可运行的 C 代码块；\n"
                    "4) `## 运行过程 / 输出结果`：说明代码为什么这样运行；\n"
                    "5) `## 易错点`：指出初学者最容易错在哪里；\n"
                    "6) `## 扩展`：补充类似写法、优化写法或考试提醒。\n"
                    "不要只给抽象概念，C语言概念题通常需要代码示例。"
                )
            if task_type == "debug":
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 问题定位`：指出最可能的错误位置或原因；\n"
                    "2) `## 修正方案`：给出修改后的关键代码；\n"
                    "3) `## 原因说明`：解释为什么会报错或出错；\n"
                    "4) `## 排查建议`：补充 1~3 条继续检查的方法。"
                )
            if task_type == "code_reading":
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 结论`：先说明这段代码整体要表达什么；\n"
                    "2) `## 代码在做什么`；\n"
                    "3) `## 关键语句说明`；\n"
                    "4) `## 运行过程 / 输出结果`；\n"
                    "5) `## 易错点`；\n"
                    "6) `## 扩展`。\n"
                    "如果代码较短，尽量逐段解释。"
                )
            if mode_id == "deepsearch":
                return (
                    "请使用 Markdown，并尽量按以下结构回答：\n"
                    "1) `## 结论`：先直接回答问题；\n"
                    "2) `## 核心概念`：讲清定义、语义和底层机制；\n"
                    "3) `## 代码示例`：提供能说明问题的 C 代码块；\n"
                    "4) `## 运行过程 / 输出结果`：解释关键语句、运行结果或内存含义；\n"
                    "5) `## 易错点`：补充常见误区、边界条件或易混概念；\n"
                    "6) `## 扩展`：补充类似写法、优化写法、相关考点或进一步思考。\n"
                    "若问题涉及区别、原理、调试或实现细节，务必把代码与解释对应起来。"
                )
            if mode_id == "instant":
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 结论`；\n"
                    "2) `## 核心概念`；\n"
                    "3) `## 代码示例`；\n"
                    "4) `## 运行过程 / 输出结果`；\n"
                    "5) `## 易错点`；\n"
                    "6) `## 扩展`（可简短）。\n"
                    "不要只给抽象定义，C语言概念题通常需要代码示例。"
                )
            return (
                "请使用 Markdown，并尽量按以下结构回答：\n"
                "1) `## 结论`；\n"
                "2) `## 核心概念`；\n"
                "3) `## 代码示例`；\n"
                "4) `## 运行过程 / 输出结果`；\n"
                "5) `## 易错点`；\n"
                "6) `## 扩展`。\n"
                "不要只给一小段概念描述，C语言回答应优先给出代码示例并解释含义。"
            )

        if subject_id == "operating_systems":
            if task_type == "comparison":
                if response_language == "en":
                    return (
                        "Use Markdown and prefer this structure:\n"
                        "1) `## Conclusion`;\n"
                        "2) `## Essence`;\n"
                        "3) `## Comparison`;\n"
                        "4) `## Exam-ready Wording`.\n"
                        "Make the differences explicit instead of giving two isolated definitions."
                    )
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 结论`；\n"
                    "2) `## 本质`；\n"
                    "3) `## 对比`；\n"
                    "4) `## 考试表达`。\n"
                    "不要只分别下定义，要把差异明确对比出来。"
                )
            if task_type == "mechanism":
                if response_language == "en":
                    return (
                        "Use Markdown and prefer this structure:\n"
                        "1) `## Conclusion`;\n"
                        "2) `## Essence`;\n"
                        "3) `## Mechanism`;\n"
                        "4) `## Diagrammatic Understanding`;\n"
                        "5) `## Exam-ready Wording`."
                    )
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 结论`；\n"
                    "2) `## 本质`；\n"
                    "3) `## 机制`；\n"
                    "4) `## 图示化理解`；\n"
                    "5) `## 考试表达`。"
                )
            if task_type == "concept":
                if response_language == "en":
                    return (
                        "Use Markdown and prefer this structure:\n"
                        "1) `## Conclusion`;\n"
                        "2) `## Essence`;\n"
                        "3) `## Mechanism`;\n"
                        "4) `## Exam-ready Wording`."
                    )
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 结论`；\n"
                    "2) `## 本质`；\n"
                    "3) `## 机制`；\n"
                    "4) `## 考试表达`。"
                )
            if response_language == "en":
                if mode_id == "deepsearch":
                    return (
                        "Use Markdown and prefer this structure:\n"
                        "1) `## Conclusion`;\n"
                        "2) `## Essence`;\n"
                        "3) `## Mechanism`;\n"
                        "4) `## Diagrammatic Understanding` when useful;\n"
                        "5) `## Comparison` when useful;\n"
                        "6) `## Exam-ready Wording`.\n"
                        "Focus on process, state transitions, design rationale, and comparisons."
                    )
                if mode_id == "instant":
                    return (
                        "Use Markdown and prefer this structure:\n"
                        "1) `## Conclusion`;\n"
                        "2) `## Essence`;\n"
                        "3) `## Mechanism`;\n"
                        "4) `## Exam-ready Wording`.\n"
                        "Avoid a single dense paragraph; make the mechanism easy to follow."
                    )
                return (
                    "Use Markdown and prefer this structure:\n"
                    "1) `## Conclusion`;\n"
                    "2) `## Essence`;\n"
                    "3) `## Mechanism`;\n"
                    "4) `## Diagrammatic Understanding` when useful;\n"
                    "5) `## Comparison` when useful;\n"
                    "6) `## Exam-ready Wording`.\n"
                    "Operating systems answers should emphasize mechanism and comparison, not just a short definition."
                )
            if mode_id == "deepsearch":
                return (
                    "请使用 Markdown，并尽量按以下结构回答：\n"
                    "1) `## 结论`；\n"
                    "2) `## 本质`；\n"
                    "3) `## 机制`；\n"
                    "4) `## 图示化理解`（适用时）；\n"
                    "5) `## 对比`（适用时）；\n"
                    "6) `## 考试表达`。\n"
                    "重点讲清流程、状态变化、设计原因和概念对比。"
                )
            if mode_id == "instant":
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 结论`；\n"
                    "2) `## 本质`；\n"
                    "3) `## 机制`；\n"
                    "4) `## 考试表达`。\n"
                    "不要只给一段笼统描述，尽量把机制讲清。"
                )
            return (
                "请使用 Markdown，并尽量按以下结构回答：\n"
                "1) `## 结论`；\n"
                "2) `## 本质`；\n"
                "3) `## 机制`；\n"
                "4) `## 图示化理解`（适用时）；\n"
                "5) `## 对比`（适用时）；\n"
                "6) `## 考试表达`。\n"
                "操作系统问题应优先解释机制、流程和概念对比。"
            )

        if subject_id == "cybersec_lab":
            if task_type == "lab_steps":
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 实验目标`；\n"
                    "2) `## 实验环境`；\n"
                    "3) `## 实验原理`；\n"
                    "4) `## 操作步骤`；\n"
                    "5) `## 结果观察`；\n"
                    "6) `## 结果分析`；\n"
                    "7) `## 常见问题`；\n"
                    "8) `## 实验结论`。\n"
                    "步骤尽量可执行，不要只讲概念。"
                )
            if task_type == "troubleshooting":
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 现象`；\n"
                    "2) `## 可能原因`；\n"
                    "3) `## 排查步骤`；\n"
                    "4) `## 修复建议`；\n"
                    "5) `## 注意事项`。"
                )
            if mode_id == "deepsearch":
                return (
                    "请使用 Markdown，并尽量按以下结构回答：\n"
                    "1) `## 实验目标`；\n"
                    "2) `## 实验环境`；\n"
                    "3) `## 实验原理`；\n"
                    "4) `## 操作步骤`；\n"
                    "5) `## 结果观察`；\n"
                    "6) `## 结果分析`；\n"
                    "7) `## 常见问题`；\n"
                    "8) `## 实验结论`。\n"
                    "如果涉及安全风险、权限、环境差异或排错，请明确写出。"
                )
            if mode_id == "instant":
                return (
                    "请使用 Markdown，并优先按以下结构回答：\n"
                    "1) `## 实验目标`；\n"
                    "2) `## 实验环境`；\n"
                    "3) `## 实验原理`；\n"
                    "4) `## 操作步骤`；\n"
                    "5) `## 结果观察`；\n"
                    "6) `## 结果分析`；\n"
                    "7) `## 常见问题`；\n"
                    "8) `## 实验结论`。\n"
                    "不要只给概念解释，实验类问题应尽量给出可操作步骤。"
                )
            return (
                "请使用 Markdown，并尽量按以下结构回答：\n"
                "1) `## 实验目标`；\n"
                "2) `## 实验环境`；\n"
                "3) `## 实验原理`；\n"
                "4) `## 操作步骤`；\n"
                "5) `## 结果观察`；\n"
                "6) `## 结果分析`；\n"
                "7) `## 常见问题`；\n"
                "8) `## 实验结论`。\n"
                "网络安全实验问题应强调步骤、前提条件和风险提醒。"
            )

        return (
            "请使用 Markdown 输出，尽量避免单段大白话，优先分点或分小节说明。"
        )

    def _apply_answer_style_to_question(
        self,
        question: str,
        *,
        user_question: str,
        subject_id: str,
        mode: str,
        response_language: str,
    ) -> str:
        language_instruction = self._response_language_instruction(response_language)
        task_type = self._detect_subject_task_type(subject_id, user_question)
        style_instruction = self._subject_answer_style_instruction(
            subject_id,
            task_type,
            mode,
            response_language,
        )
        text = str(question or "").rstrip()
        instructions = (
            f"[Answer language requirement]\n{language_instruction}\n\n"
            f"[Detected task type]\n{task_type}\n\n"
            f"[Answer format requirement]\n{style_instruction}"
        )
        if not text:
            return instructions
        return f"{text}\n\n{instructions}"
