from __future__ import annotations

import unittest

from webapp_core.graph_service import Neo4jGraphService


class _FakeGraphService(Neo4jGraphService):
    @property
    def configured(self) -> bool:
        return True

    def _run_read(self, query: str, **params: object) -> list[dict[str, object]]:
        if "labels(n)[0]" in query:
            return [{"label": "Entity", "count": 2}, {"label": "Chunk", "count": 1}]
        if "MATCH (s:Subject)" in query:
            return [{"id": "C_program", "name": "C_program"}]
        if "MATCH (e:Entity)" in query and "RETURN e.id AS id" in query:
            return [
                {
                    "id": "C_program:printf",
                    "label": "printf",
                    "subjectId": "C_program",
                    "type": "function",
                    "description": "格式化输出函数",
                    "score": 1.0,
                }
            ]
        if "MATCH (center:Entity)" in query:
            return [
                {
                    "nodes": [
                        {
                            "id": "C_program:printf",
                            "label": "printf",
                            "subjectId": "C_program",
                            "type": "function",
                            "hitType": "direct",
                            "score": 1.0,
                            "description": "格式化输出函数",
                            "sourceIds": ["chunk-1"],
                        }
                    ],
                    "edges": [
                        {
                            "id": "edge-1",
                            "source": "C_program:printf",
                            "target": "C_program:stdio",
                            "label": "BELONGS_TO",
                            "weight": 1.0,
                            "description": "属于 stdio",
                            "keywords": "library",
                        }
                    ],
                }
            ]
        if "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)" in query:
            return [
                {
                    "id": "C_program:chunk-1",
                    "chunkId": "chunk-1",
                    "subjectId": "C_program",
                    "preview": "printf 示例",
                    "content": "printf 示例内容",
                    "tokens": 12,
                    "filePath": "demo.txt",
                    "rawChunkId": "chunk-1",
                }
            ]
        return []


class GraphServiceTests(unittest.TestCase):
    def test_health_returns_counts_and_subjects(self) -> None:
        service = _FakeGraphService(uri="bolt://demo", password="pw")

        result = service.health()

        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["Entity"], 2)
        self.assertEqual(result["subjects"][0]["id"], "C_program")

    def test_local_subgraph_normalizes_nodes_edges_and_chunks(self) -> None:
        service = _FakeGraphService(uri="bolt://demo", password="pw")

        result = service.local_subgraph(
            subject_ids=["C_program"],
            center_entity_ids=["C_program:printf"],
            depth=1,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["nodes"][0]["id"], "C_program:printf")
        self.assertEqual(result["nodes"][0]["hitType"], "direct")
        self.assertEqual(result["edges"][0]["source"], "C_program:printf")
        self.assertEqual(result["chunks"][0]["chunkId"], "chunk-1")

    def test_unconfigured_service_reports_unavailable(self) -> None:
        service = Neo4jGraphService(uri="", password="")

        result = service.local_subgraph(query="printf")

        self.assertFalse(result["ok"])
        self.assertIn("neo4j", result["error"].lower())

    def test_query_terms_drop_course_generic_words(self) -> None:
        terms = Neo4jGraphService._query_terms(
            "C language memory management, Dangling pointer errors"
        )

        self.assertIn("memory", terms)
        self.assertIn("dangling", terms)
        self.assertIn("pointer", terms)
        self.assertNotIn("c", terms)
        self.assertNotIn("language", terms)
        self.assertNotIn("errors", terms)


if __name__ == "__main__":
    unittest.main()
