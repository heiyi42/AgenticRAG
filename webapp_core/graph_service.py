from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


try:  # Neo4j is an optional runtime dependency for the visualization surface.
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - exercised when optional dependency is absent.
    GraphDatabase = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "code",
    "error",
    "errors",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "language",
    "languages",
    "of",
    "on",
    "or",
    "program",
    "programming",
    "the",
    "to",
    "use",
    "using",
    "what",
    "when",
    "where",
    "why",
    "with",
    "什么",
    "如何",
    "为什么",
    "请问",
    "解释",
    "分析",
    "代码",
    "程序",
    "语言",
    "问题",
}


class Neo4jGraphService:
    def __init__(
        self,
        *,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = (uri if uri is not None else os.getenv("NEO4J_URI") or "").strip()
        self.username = (
            username
            if username is not None
            else os.getenv("NEO4J_USERNAME")
            or os.getenv("NEO4J_USER")
            or "neo4j"
        ).strip()
        self.password = (
            password
            if password is not None
            else os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS") or ""
        ).strip()
        self.database = (
            database if database is not None else os.getenv("NEO4J_DATABASE") or ""
        ).strip() or None
        self._driver: Any = None

    @property
    def configured(self) -> bool:
        return bool(GraphDatabase is not None and self.uri and self.password)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _driver_or_error(self) -> tuple[Any | None, str | None]:
        if GraphDatabase is None:
            return None, "neo4j package is not installed; install project tools extras."
        if not self.uri:
            return None, "NEO4J_URI is not configured."
        if not self.password:
            return None, "NEO4J_PASSWORD is not configured."
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
        return self._driver, None

    def _run_read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        driver, error = self._driver_or_error()
        if error:
            raise RuntimeError(error)
        with driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def health(self) -> dict[str, Any]:
        if not self.configured:
            _, error = self._driver_or_error()
            return {
                "ok": False,
                "configured": False,
                "error": error or "Neo4j is not configured.",
                "subjects": [],
                "counts": {},
            }
        try:
            rows = self._run_read(
                """
                MATCH (n)
                WHERE n:Subject OR n:Entity OR n:Document OR n:Chunk
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY label
                """
            )
            subjects = self._run_read(
                """
                MATCH (s:Subject)
                RETURN s.id AS id, coalesce(s.name, s.id) AS name
                ORDER BY id
                """
            )
            return {
                "ok": True,
                "configured": True,
                "error": "",
                "subjects": subjects,
                "counts": {str(row["label"]): int(row["count"]) for row in rows},
            }
        except Exception as exc:
            return {
                "ok": False,
                "configured": True,
                "error": f"{type(exc).__name__}: {exc}",
                "subjects": [],
                "counts": {},
            }

    def search_entities(
        self,
        *,
        subject_id: str = "",
        query: str = "",
        limit: int = 8,
    ) -> dict[str, Any]:
        q = str(query or "").strip()
        if not q:
            return {"ok": True, "entities": []}
        terms = self._query_terms(q)
        if not terms:
            return {"ok": True, "entities": []}
        safe_limit = max(1, min(30, int(limit or 8)))
        try:
            rows = self._run_read(
                """
                MATCH (e:Entity)
                WHERE ($subject_id = "" OR e.subject = $subject_id)
                WITH e,
                     toLower(coalesce(e.name, "") + " " + coalesce(e.displayName, "")) AS nameText,
                     toLower(coalesce(e.description, "") + " " + coalesce(e.entity_type, "")) AS detailText,
                     $terms AS terms
                WITH e,
                     nameText,
                     [term IN terms WHERE nameText CONTAINS term] AS nameHits,
                     [term IN terms WHERE detailText CONTAINS term] AS detailHits
                WHERE size(nameHits) > 0 OR size(detailHits) > 0 OR nameText CONTAINS $q
                RETURN e.id AS id,
                       coalesce(e.name, e.displayName, e.id) AS label,
                       e.subject AS subjectId,
                       coalesce(e.entity_type, "entity") AS type,
                       coalesce(e.description, "") AS description,
                       toFloat(size(nameHits) * 3 + size(detailHits)) +
                         CASE WHEN nameText CONTAINS $q THEN 3.0 ELSE 0.0 END AS score
                ORDER BY score DESC, size(coalesce(e.name, "")) ASC, label ASC
                LIMIT $limit
                """,
                subject_id=str(subject_id or "").strip(),
                q=q.lower(),
                terms=terms,
                limit=safe_limit,
            )
            return {"ok": True, "entities": [self._normalize_entity(row) for row in rows]}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "entities": []}

    def _search_entities_from_chunks(
        self,
        *,
        subject_ids: list[str],
        query: str,
        limit: int,
    ) -> dict[str, Any]:
        terms = self._query_terms(query)
        if not terms:
            return {"ok": True, "entities": []}
        safe_limit = max(1, min(30, int(limit or 8)))
        try:
            rows = self._run_read(
                """
                MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
                WHERE (size($subject_ids) = 0 OR e.subject IN $subject_ids)
                WITH e,
                     toLower(
                       coalesce(c.content_preview, "") + " " +
                       coalesce(c.content, "")
                     ) AS chunkText,
                     $terms AS terms
                WITH e, [term IN terms WHERE chunkText CONTAINS term] AS hits
                WITH e, sum(size(hits)) AS score
                WHERE score > 0
                RETURN e.id AS id,
                       coalesce(e.name, e.displayName, e.id) AS label,
                       e.subject AS subjectId,
                       coalesce(e.entity_type, "entity") AS type,
                       coalesce(e.description, "") AS description,
                       toFloat(score) AS score
                ORDER BY score DESC, size(coalesce(e.name, "")) ASC, label ASC
                LIMIT $limit
                """,
                subject_ids=subject_ids,
                terms=terms,
                limit=safe_limit,
            )
            return {"ok": True, "entities": [self._normalize_entity(row) for row in rows]}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "entities": []}

    def entity_chunks(self, *, entity_id: str, limit: int = 6) -> dict[str, Any]:
        safe_limit = max(1, min(20, int(limit or 6)))
        try:
            rows = self._run_read(
                """
                MATCH (e:Entity {id: $entity_id})-[r:MENTIONED_IN]->(c:Chunk)
                RETURN c.id AS id,
                       coalesce(c.chunk_id, c.id) AS chunkId,
                       c.subject AS subjectId,
                       coalesce(c.content_preview, left(coalesce(c.content, ""), 240)) AS preview,
                       coalesce(c.content, "") AS content,
                       c.tokens AS tokens,
                       c.file_path AS filePath,
                       r.chunk_raw_id AS rawChunkId
                ORDER BY chunkId
                LIMIT $limit
                """,
                entity_id=str(entity_id or "").strip(),
                limit=safe_limit,
            )
            return {"ok": True, "chunks": [self._normalize_chunk(row) for row in rows]}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "chunks": []}

    def local_subgraph(
        self,
        *,
        subject_ids: list[str] | None = None,
        query: str = "",
        center_entity_ids: list[str] | None = None,
        depth: int = 1,
        limit: int = 80,
    ) -> dict[str, Any]:
        if not self.configured:
            _, error = self._driver_or_error()
            return {
                "ok": False,
                "error": error or "Neo4j is not configured.",
                "nodes": [],
                "edges": [],
                "chunks": [],
                "centerEntityIds": [],
                "subjectIds": subject_ids or [],
            }
        safe_subjects = [str(item).strip() for item in (subject_ids or []) if str(item).strip()]
        safe_centers = [
            str(item).strip()
            for item in (center_entity_ids or [])
            if str(item).strip()
        ]
        if not safe_centers and query:
            search_subject = safe_subjects[0] if len(safe_subjects) == 1 else ""
            search_result = self.search_entities(
                subject_id=search_subject,
                query=query,
                limit=5,
            )
            if not search_result.get("ok", False):
                return {
                    "ok": False,
                    "error": str(search_result.get("error", "Neo4j search failed.")),
                    "nodes": [],
                    "edges": [],
                    "chunks": [],
                    "centerEntityIds": [],
                    "subjectIds": safe_subjects,
                }
            safe_centers = [
                str(item["id"])
                for item in search_result.get("entities", [])
                if isinstance(item, dict) and item.get("id")
            ]
            if not safe_centers:
                chunk_search_result = self._search_entities_from_chunks(
                    subject_ids=safe_subjects,
                    query=query,
                    limit=5,
                )
                if not chunk_search_result.get("ok", False):
                    return {
                        "ok": False,
                        "error": str(
                            chunk_search_result.get("error", "Neo4j chunk search failed.")
                        ),
                        "nodes": [],
                        "edges": [],
                        "chunks": [],
                        "centerEntityIds": [],
                        "subjectIds": safe_subjects,
                    }
                safe_centers = [
                    str(item["id"])
                    for item in chunk_search_result.get("entities", [])
                    if isinstance(item, dict) and item.get("id")
                ]
        if not safe_centers:
            return {
                "ok": True,
                "nodes": [],
                "edges": [],
                "chunks": [],
                "centerEntityIds": [],
                "subjectIds": safe_subjects,
            }

        safe_depth = 2 if int(depth or 1) >= 2 else 1
        safe_limit = max(10, min(200, int(limit or 80)))
        query_text = self._subgraph_query(depth=safe_depth)
        try:
            rows = self._run_read(
                query_text,
                center_ids=safe_centers,
                subject_ids=safe_subjects,
                limit=safe_limit,
            )
            row = rows[0] if rows else {}
            nodes = [self._normalize_node(item) for item in row.get("nodes", [])]
            edges = [self._normalize_edge(item) for item in row.get("edges", [])]
            chunk_rows = self._chunks_for_nodes([node["id"] for node in nodes[:8]])
            return {
                "ok": True,
                "nodes": nodes,
                "edges": edges,
                "chunks": chunk_rows,
                "centerEntityIds": safe_centers,
                "subjectIds": safe_subjects,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "nodes": [],
                "edges": [],
                "chunks": [],
                "centerEntityIds": safe_centers,
                "subjectIds": safe_subjects,
            }

    def _chunks_for_nodes(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        rows = self._run_read(
            """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.id IN $entity_ids
            RETURN DISTINCT c.id AS id,
                   coalesce(c.chunk_id, c.id) AS chunkId,
                   c.subject AS subjectId,
                   coalesce(c.content_preview, left(coalesce(c.content, ""), 240)) AS preview,
                   coalesce(c.content, "") AS content,
                   c.tokens AS tokens,
                   c.file_path AS filePath,
                   "" AS rawChunkId
            LIMIT 10
            """,
            entity_ids=entity_ids,
        )
        return [self._normalize_chunk(row) for row in rows]

    @staticmethod
    def _subgraph_query(*, depth: int) -> str:
        hop_pattern = "[*1..2]" if depth >= 2 else "[*1..1]"
        return f"""
        MATCH (center:Entity)
        WHERE center.id IN $center_ids
          AND (size($subject_ids) = 0 OR center.subject IN $subject_ids)
        OPTIONAL MATCH p=(center)-{hop_pattern}-(neighbor:Entity)
        WHERE neighbor.subject = center.subject
        WITH collect(DISTINCT center) AS centers,
             collect(DISTINCT neighbor) AS neighbors,
             [path IN collect(p) WHERE path IS NOT NULL] AS paths
        WITH (centers + [n IN neighbors WHERE n IS NOT NULL])[0..$limit] AS rawNodes,
             paths,
             [c IN centers | c.id] AS centerIds
        CALL (paths) {{
          UNWIND paths AS path
          UNWIND relationships(path) AS rel
          RETURN collect(DISTINCT rel)[0..$limit] AS rels
        }}
        RETURN [
          n IN rawNodes | {{
            id: n.id,
            label: coalesce(n.name, n.displayName, n.id),
            subjectId: n.subject,
            type: coalesce(n.entity_type, "entity"),
            hitType: CASE WHEN n.id IN centerIds THEN "direct" ELSE "related" END,
            score: CASE WHEN n.id IN centerIds THEN 1.0 ELSE 0.45 END,
            description: coalesce(n.description, ""),
            sourceIds: coalesce(n.source_ids, [])
          }}
        ] AS nodes,
        [
          r IN rels | {{
            id: coalesce(r.id, elementId(r)),
            source: startNode(r).id,
            target: endNode(r).id,
            label: type(r),
            weight: coalesce(r.weight, 0.0),
            description: coalesce(r.description, ""),
            keywords: coalesce(r.keywords, "")
          }}
        ] AS edges
        """

    @staticmethod
    def _normalize_entity(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id", "")),
            "label": str(row.get("label", "")),
            "subjectId": str(row.get("subjectId", "")),
            "type": str(row.get("type", "entity") or "entity"),
            "description": str(row.get("description", "")),
            "score": float(row.get("score") or 0.0),
        }

    @staticmethod
    def _normalize_node(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id", "")),
            "label": str(row.get("label", "")),
            "subjectId": str(row.get("subjectId", "")),
            "type": str(row.get("type", "entity") or "entity"),
            "hitType": str(row.get("hitType", "none") or "none"),
            "score": float(row.get("score") or 0.0),
            "metadata": {
                "description": str(row.get("description", "")),
                "sourceIds": list(row.get("sourceIds") or []),
            },
        }

    @staticmethod
    def _normalize_edge(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id", "")),
            "source": str(row.get("source", "")),
            "target": str(row.get("target", "")),
            "label": str(row.get("label", "")),
            "weight": float(row.get("weight") or 0.0),
            "metadata": {
                "description": str(row.get("description", "")),
                "keywords": str(row.get("keywords", "")),
            },
        }

    @staticmethod
    def _normalize_chunk(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id", "")),
            "chunkId": str(row.get("chunkId", "")),
            "subjectId": str(row.get("subjectId", "")),
            "preview": str(row.get("preview", "")),
            "content": str(row.get("content", "")),
            "tokens": row.get("tokens", ""),
            "filePath": str(row.get("filePath", "")),
            "rawChunkId": str(row.get("rawChunkId", "")),
        }

    @staticmethod
    def _query_terms(query: str, *, max_terms: int = 18) -> list[str]:
        text = str(query or "").lower()
        raw_terms = re.findall(r"[a-z][a-z0-9_+#]*|\d+|[\u4e00-\u9fff]{2,}", text)
        terms: list[str] = []
        seen: set[str] = set()

        def add(term: str) -> None:
            clean = term.strip().lower()
            if not clean or clean in _STOPWORDS or clean in seen:
                return
            if re.fullmatch(r"[a-z0-9_+#]+", clean) and len(clean) < 2:
                return
            seen.add(clean)
            terms.append(clean)

        for raw in raw_terms:
            add(raw)
            if re.fullmatch(r"[\u4e00-\u9fff]+", raw) and len(raw) >= 3:
                for size in (4, 3, 2):
                    for index in range(0, len(raw) - size + 1):
                        add(raw[index : index + size])
                        if len(terms) >= max_terms:
                            return terms[:max_terms]
            if len(terms) >= max_terms:
                return terms[:max_terms]

        return terms[:max_terms]
