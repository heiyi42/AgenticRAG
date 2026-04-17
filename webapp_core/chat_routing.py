from __future__ import annotations

import asyncio
import hashlib
import re
import time
from pathlib import Path
from typing import Any

import models.auto as auto
from pydantic import BaseModel, Field

from . import config as cfg


class RetrievalGateDecision(BaseModel):
    need_retrieval: bool = Field(description="是否需要检索")
    confidence: float = Field(ge=0.0, le=1.0, description="判断置信度")
    reason: str = Field(description="简短理由")
    kb_relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="问题与知识库语料语义相关性评分",
    )
    direct_answerability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="不依赖知识库可直接回答的可行性评分",
    )


class SubjectRouteDecision(BaseModel):
    primary_subject: str = Field(description="主学科ID")
    cross_subject: bool = Field(description="是否跨学科")
    confidence: float = Field(ge=0.0, le=1.0, description="学科路由置信度")
    reason: str = Field(description="简短理由")
    c_program_score: float = Field(ge=0.0, le=1.0, default=0.0)
    operating_systems_score: float = Field(ge=0.0, le=1.0, default=0.0)
    cybersec_lab_score: float = Field(ge=0.0, le=1.0, default=0.0)


class ChatRoutingMixin:
    @classmethod
    def _subject_label(cls, subject_id: str) -> str:
        return cls.SUBJECT_LABELS.get(subject_id, subject_id)

    def _build_subject_catalog(self) -> dict[str, dict[str, str]]:
        storage_root = Path(cfg.WEB_STORAGE_ROOT)
        catalog: dict[str, dict[str, str]] = {}
        for subject_id, label in self.SUBJECT_LABELS.items():
            working_dir = (storage_root / subject_id).resolve()
            catalog[subject_id] = {
                "id": subject_id,
                "label": label,
                "working_dir": str(working_dir),
            }
        return catalog

    def _subject_gate_scope_text(self, subject_ids: list[str]) -> str:
        short_scope = {
            "C_program": (
                "C_program（C语言）：C 语法、指针/数组/结构体、内存管理、函数、"
                "文件 IO 等课程基础内容。"
            ),
            "operating_systems": (
                "operating_systems（操作系统）：进程/线程、调度、同步互斥、"
                "内存管理、分页、中断、系统调用、文件系统等课程内容。"
            ),
            "cybersec_lab": (
                "cybersec_lab（网络安全实验）：实验环境、协议分析、加密认证、"
                "访问控制、常见漏洞与防护、实验工具使用等课程内容。"
            ),
        }
        return "\n".join(
            short_scope.get(
                subject_id,
                f"{subject_id}（{self._subject_label(subject_id)}）：课程相关内容。",
            )
            for subject_id in subject_ids
        )

    @staticmethod
    def _normalize_gate_text(text: str, max_len: int = 800) -> str:
        compact = re.sub(r"\s+", " ", str(text or "")).strip()
        return compact[:max_len]

    @staticmethod
    def _compact_context_for_gate(text: str, max_len: int = 520) -> str:
        compact = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(compact) <= max_len:
            return compact
        head = compact[: max(120, max_len // 2)].rstrip()
        tail = compact[-max(90, max_len // 3) :].lstrip()
        return f"{head} ... {tail}"

    @classmethod
    def _subject_scores_from_obj(cls, obj: SubjectRouteDecision) -> dict[str, float]:
        return {
            subject_id: auto._clamp_confidence(getattr(obj, field_name, 0.0))
            for subject_id, field_name in cls.SUBJECT_SCORE_FIELDS.items()
        }

    @classmethod
    def _rank_subject_scores(
        cls, scores: dict[str, float]
    ) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    @classmethod
    def _pick_subjects_by_keywords(cls, text: str) -> dict[str, float]:
        normalized = re.sub(r"\s+", "", str(text or "")).lower()
        if not normalized:
            return {subject_id: 0.0 for subject_id in cls.SUBJECT_LABELS}

        scores: dict[str, float] = {}
        for subject_id, keywords in cls.SUBJECT_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword.lower() in normalized)
            scores[subject_id] = min(1.0, hits * 0.2)
        return scores

    def _subject_route_from_scores(
        self,
        scores: dict[str, float],
        *,
        confidence: float,
        reason: str,
        requested_subjects: list[str] | None = None,
    ) -> dict[str, Any]:
        ranked = self._rank_subject_scores(scores)
        primary_subject, primary_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        cross_subject = primary_score >= 0.30 and second_score >= 0.30
        return {
            "scores": scores,
            "ranked": ranked,
            "primary_subject": primary_subject,
            "cross_subject": cross_subject,
            "confidence": auto._clamp_confidence(confidence),
            "reason": reason,
            "max_score": primary_score,
            "requested_subjects": list(requested_subjects or []),
        }

    def _subject_route_from_explicit_subjects(
        self, subject_ids: list[str]
    ) -> dict[str, Any]:
        scores = {subject_id: 0.0 for subject_id in self.SUBJECT_LABELS}
        base = 1.0
        for index, subject_id in enumerate(subject_ids):
            scores[subject_id] = max(0.1, base - index * 0.05)
        return self._subject_route_from_scores(
            scores,
            confidence=1.0,
            reason="用户显式指定学科",
            requested_subjects=subject_ids,
        )

    def normalize_requested_subjects(self, raw_subjects: Any) -> list[str]:
        if raw_subjects is None:
            return []
        if isinstance(raw_subjects, str):
            candidates = re.split(r"[\s,，|/]+", raw_subjects)
        elif isinstance(raw_subjects, list):
            candidates = raw_subjects
        else:
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        alias_map = {
            "c": "C_program",
            "c_program": "C_program",
            "cprogram": "C_program",
            "c语言": "C_program",
            "os": "operating_systems",
            "operating_systems": "operating_systems",
            "操作系统": "operating_systems",
            "cybersec": "cybersec_lab",
            "cybersec_lab": "cybersec_lab",
            "网络安全": "cybersec_lab",
            "网络安全实验": "cybersec_lab",
        }
        for item in candidates:
            value = str(item or "").strip()
            if not value:
                continue
            subject_id = alias_map.get(value.lower(), value)
            if subject_id in self.subject_catalog and subject_id not in seen:
                seen.add(subject_id)
                normalized.append(subject_id)
        return normalized

    @staticmethod
    def _normalize_auto_complexity(value: Any) -> str:
        return "simple" if str(value or "").strip().lower() == "simple" else "complex"

    async def decide_subject_route(
        self,
        *,
        user_question: str,
        augmented_question: str,
        mode: str,
        timeout_s: int,
        requested_subjects: list[str] | None = None,
    ) -> dict[str, Any]:
        explicit_subjects = [s for s in (requested_subjects or []) if s in self.subject_catalog]
        if explicit_subjects:
            return self._subject_route_from_explicit_subjects(explicit_subjects)

        keyword_scores = self._pick_subjects_by_keywords(
            f"{user_question}\n{augmented_question}"
        )
        ranked_keywords = self._rank_subject_scores(keyword_scores)
        if ranked_keywords[0][1] >= 0.6:
            return self._subject_route_from_scores(
                keyword_scores,
                confidence=0.9,
                reason="关键词命中学科特征",
                requested_subjects=explicit_subjects,
            )

        route_timeout = max(
            1,
            min(int(timeout_s), max(1, int(cfg.WEB_RETRIEVAL_GATE_TIMEOUT_S))),
        )
        subject_scope = self._subject_gate_scope_text(list(self.subject_catalog.keys()))
        prompt = (
            "你是第一个网关：多学科课程路由器。\n"
            "请判断用户问题与三个课程知识库的相关度，并识别是否跨学科。\n"
            "三个候选课程范围如下（仅供边界判断，不是完整章节画像）：\n"
            f"{subject_scope}\n\n"
            "规则：\n"
            "1) 给出 primary_subject、cross_subject、confidence、reason。\n"
            "2) 同时给出三个分数：c_program_score、operating_systems_score、cybersec_lab_score，范围 0~1。\n"
            "3) 栈溢出、缓冲区溢出、内存破坏这类问题可能同时关联 C语言、操作系统、网络安全实验。\n"
            "4) 若问题明显不属于这三类课程，三个分数都应较低。\n"
            "5) primary_subject 必须是 C_program / operating_systems / cybersec_lab 三者之一。\n"
            "只输出结构化字段。\n\n"
            f"当前模式：{mode}\n"
            f"用户问题：{user_question}\n"
            f"上下文增强问题：{self._compact_context_for_gate(augmented_question, max_len=520)}"
        )
        try:
            obj = await asyncio.wait_for(
                self.llm_subject_router_struct.ainvoke(prompt),
                timeout=route_timeout,
            )
            scores = self._subject_scores_from_obj(obj)
            return self._subject_route_from_scores(
                scores,
                confidence=getattr(obj, "confidence", 0.5),
                reason=str(getattr(obj, "reason", "") or "").strip() or "无",
                requested_subjects=explicit_subjects,
            )
        except Exception as e:
            return self._subject_route_from_scores(
                keyword_scores,
                confidence=0.0,
                reason=f"学科路由失败，回退关键词规则: {e}",
                requested_subjects=explicit_subjects,
            )

    @staticmethod
    def _subjects_for_instant(subject_route: dict[str, Any]) -> list[str]:
        ranked = list(subject_route.get("ranked", []))
        if not ranked:
            return ["operating_systems"]
        return [ranked[0][0]]

    @staticmethod
    def _subjects_for_deep(subject_route: dict[str, Any]) -> list[str]:
        ranked = list(subject_route.get("ranked", []))
        selected = [subject_id for subject_id, score in ranked if score >= 0.28]
        return selected[:3] or ["operating_systems"]

    @staticmethod
    def _subjects_for_auto(subject_route: dict[str, Any]) -> list[str]:
        requested = list(subject_route.get("requested_subjects", []))
        if len(requested) >= 2:
            return requested[:2]
        ranked = list(subject_route.get("ranked", []))
        if not ranked:
            return ["operating_systems"]
        selected = [ranked[0][0]]
        if len(ranked) > 1:
            top_score = ranked[0][1]
            second_score = ranked[1][1]
            if top_score >= 0.35 and second_score >= 0.35 and (top_score - second_score) <= 0.15:
                selected.append(ranked[1][0])
        return selected

    @staticmethod
    def _score_route_need_retrieval(
        *,
        kb_relevance: float,
        direct_answerability: float,
        model_need_retrieval: bool,
    ) -> tuple[bool, str]:
        kb = auto._clamp_confidence(kb_relevance)
        direct = auto._clamp_confidence(direct_answerability)
        delta = kb - direct
        kb_th = auto._clamp_confidence(cfg.WEB_RETRIEVAL_GATE_KB_THRESHOLD)
        direct_th = auto._clamp_confidence(cfg.WEB_RETRIEVAL_GATE_DIRECT_THRESHOLD)
        margin = max(0.0, float(cfg.WEB_RETRIEVAL_GATE_MARGIN))

        if kb >= kb_th and delta >= -0.08:
            return True, "score:kb_high"
        if direct >= direct_th and kb < (kb_th - 0.08):
            return False, "score:direct_high"
        if delta >= margin:
            return True, "score:delta_to_kb"
        if delta <= -margin:
            return False, "score:delta_to_direct"
        return bool(model_need_retrieval), "score:gray_follow_model"

    @staticmethod
    def _normalize_for_exact_match(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        value = re.sub(r"\s+", "", value)
        value = value.strip("`*_~\"'“”‘’[](){}<>")
        value = re.sub(r"[。！？!?，,、；;：:\.\-—\s]+$", "", value)
        return value.lower()

    @classmethod
    def _strip_leading_question_echo(cls, answer: str, question: str) -> str:
        raw_answer = str(answer or "")
        raw_question = str(question or "")
        if not raw_answer.strip() or not raw_question.strip():
            return raw_answer

        question_norm = cls._normalize_for_exact_match(raw_question)
        if not question_norm:
            return raw_answer

        lines = raw_answer.splitlines()
        scan_limit = min(len(lines), 4)
        for idx in range(scan_limit):
            line = lines[idx].strip()
            if not line:
                continue
            candidate = line
            candidate = re.sub(r"^#{1,6}\s*", "", candidate)
            candidate = re.sub(
                r"^(问题|用户问题|提问|question|q)\s*[:：]\s*",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
            candidate = candidate.strip("`*_~\"'“”‘’[](){}<>")
            candidate_norm = cls._normalize_for_exact_match(candidate)
            if candidate_norm and candidate_norm == question_norm:
                next_idx = idx + 1
                while next_idx < len(lines) and not lines[next_idx].strip():
                    next_idx += 1
                return "\n".join(lines[next_idx:]).lstrip()
            break
        return raw_answer

    def _build_gate_cache_key(
        self,
        *,
        subject_ids: list[str],
        subject_profile: str,
        user_question: str,
        augmented_question: str,
        mode: str,
        response_language: str,
    ) -> str:
        payload = "\n".join(
            [
                self.GATE_PROMPT_VERSION,
                ",".join(subject_ids),
                str(mode or "").strip().lower(),
                str(response_language or "zh").strip().lower(),
                self._normalize_gate_text(user_question, 600),
                self._normalize_gate_text(augmented_question, 900),
                self._normalize_gate_text(subject_profile, 900),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _get_cached_gate_decision(
        self,
        cache_key: str,
    ) -> tuple[bool, float, str] | None:
        ttl_s = int(cfg.WEB_RETRIEVAL_GATE_CACHE_TTL_S)
        if ttl_s <= 0:
            return None
        now = time.time()
        with self._retrieval_gate_cache_lock:
            cached = self._retrieval_gate_cache.get(cache_key)
            if not cached:
                return None
            need_retrieval, confidence, reason, ts = cached
            if now - ts > ttl_s:
                self._retrieval_gate_cache.pop(cache_key, None)
                return None
        return need_retrieval, confidence, reason

    def _set_cached_gate_decision(
        self,
        cache_key: str,
        *,
        need_retrieval: bool,
        confidence: float,
        reason: str,
    ) -> None:
        ttl_s = int(cfg.WEB_RETRIEVAL_GATE_CACHE_TTL_S)
        if ttl_s <= 0:
            return
        max_entries = max(8, int(cfg.WEB_RETRIEVAL_GATE_CACHE_MAX_ENTRIES))
        now = time.time()
        with self._retrieval_gate_cache_lock:
            self._retrieval_gate_cache[cache_key] = (
                bool(need_retrieval),
                float(confidence),
                str(reason or ""),
                now,
            )
            if len(self._retrieval_gate_cache) <= max_entries:
                return

            expired_keys = [
                k
                for k, (_, _, _, ts) in self._retrieval_gate_cache.items()
                if now - ts > ttl_s
            ]
            for key in expired_keys:
                self._retrieval_gate_cache.pop(key, None)

            overflow = len(self._retrieval_gate_cache) - max_entries
            if overflow <= 0:
                return
            oldest_keys = sorted(
                self._retrieval_gate_cache.items(),
                key=lambda item: item[1][3],
            )[:overflow]
            for key, _ in oldest_keys:
                self._retrieval_gate_cache.pop(key, None)

    async def decide_need_retrieval(
        self,
        *,
        subject_ids: list[str],
        user_question: str,
        augmented_question: str,
        mode: str,
        timeout_s: int,
        response_language: str = "zh",
    ) -> tuple[bool, float, str]:
        gate_timeout = max(
            1,
            min(
                int(timeout_s),
                max(1, int(cfg.WEB_RETRIEVAL_GATE_TIMEOUT_S)),
            ),
        )
        subject_scope = self._subject_gate_scope_text(subject_ids)
        cache_key = self._build_gate_cache_key(
            subject_ids=subject_ids,
            subject_profile=subject_scope,
            user_question=user_question,
            augmented_question=augmented_question,
            mode=mode,
            response_language=response_language,
        )
        cached = self._get_cached_gate_decision(cache_key)
        if cached is not None:
            return cached

        compact_augmented = self._compact_context_for_gate(augmented_question, max_len=520)
        prompt = (
            "你是检索必要性判断器。请判断当前问题是否必须先做知识库检索。\n"
            f"知识库背景：{cfg.WEB_KB_SCOPE_DESC}\n"
            "当前候选课程范围（仅供边界判断，不是完整章节画像）：\n"
            f"{subject_scope}\n\n"
            "规则：\n"
            "1) 必须语义判断，不做关键词匹配。\n"
            "2) 先给两个分数：kb_relevance 与 direct_answerability（0~1）。\n"
            "3) 再给 need_retrieval、confidence、reason。\n"
            "4) 当前模式（auto/instant/deepsearch）不是检索判定依据。\n"
            "5) 只有当问题明确属于 C语言、操作系统、网络安全实验 这三类课程语料时，才允许提高 kb_relevance 并考虑 need_retrieval=true。\n"
            "6) 只要问题不属于上述三类课程内容，必须优先判为免检索直答：direct_answerability 应显著高于 kb_relevance，need_retrieval=false。\n"
            "7) 闲聊、寒暄、情感表达、常识问答、改写润色、开放式建议、与三类课程无关的一般问答，都应走直答。\n"
            "8) 只有涉及课程概念、章节细节、实验步骤、术语定义、原文证据、课内知识点时，才应更高 kb_relevance。\n"
            "只输出结构化字段 need_retrieval/confidence/reason/kb_relevance/direct_answerability。\n\n"
            f"当前模式：{mode}\n"
            f"用户问题：{user_question}\n"
            f"上下文增强问题：{compact_augmented}"
        )
        try:
            obj = await asyncio.wait_for(
                self.llm_retrieval_gate_struct.ainvoke(prompt),
                timeout=gate_timeout,
            )
            kb_relevance = auto._clamp_confidence(getattr(obj, "kb_relevance", 0.5))
            direct_answerability = auto._clamp_confidence(
                getattr(obj, "direct_answerability", 0.5)
            )
            need_retrieval, route_tag = self._score_route_need_retrieval(
                kb_relevance=kb_relevance,
                direct_answerability=direct_answerability,
                model_need_retrieval=bool(obj.need_retrieval),
            )
            confidence = auto._clamp_confidence(obj.confidence)
            reason = str(getattr(obj, "reason", "") or "").strip() or "无"
            reason = (
                f"{reason} | {route_tag} | kb={kb_relevance:.2f}, direct={direct_answerability:.2f}"
            )
            if confidence < cfg.WEB_RETRIEVAL_GATE_MIN_CONFIDENCE:
                final = (
                    need_retrieval,
                    confidence,
                    f"低置信度，跟随模型决策：{reason}",
                )
            else:
                final = (need_retrieval, confidence, reason)
            self._set_cached_gate_decision(
                cache_key,
                need_retrieval=final[0],
                confidence=final[1],
                reason=final[2],
            )
            return final
        except Exception as e:
            return True, 0.0, f"检索网关失败，保守走检索: {e}"

    def _build_direct_answer_prompt(
        self,
        *,
        user_question: str,
        augmented_question: str,
        mode: str,
        thread_id: str,
        response_language: str = "zh",
    ) -> str:
        return (
            "你是问答助手。当前处于免检索直答阶段。\n"
            "请仅基于用户问题、当前会话上下文和你已有通用知识回答，不调用外部检索。\n"
            "要求：\n"
            "1) 使用 Markdown 输出；\n"
            "2) 如无法确定事实或涉及时效/来源，必须明确说明不确定并建议改走检索；\n"
            "3) 不要编造来源；\n"
            "4) 不要复述用户原问题，不要把原问题当标题。\n\n"
            f"回答语言要求：{self._response_language_instruction(response_language)}\n\n"
            f"当前模式：{mode}\n"
            f"thread_id：{thread_id}\n"
            f"用户问题：{user_question}\n"
            f"上下文增强问题：{augmented_question}"
        )
