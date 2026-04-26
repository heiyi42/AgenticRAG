import argparse
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "storage"
GRAPHML_NAME = "graph_chunk_entity_relation.graphml"
SEP = "<SEP>"
GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}

DEFAULT_BATCH_SIZE_NODES = 500
DEFAULT_BATCH_SIZE_EDGES = 200


load_dotenv(PROJECT_ROOT / ".env")


def env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def clean_value(value):
    if value is None:
        return ""
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def coerce_graphml_value(value, value_type):
    value = clean_value(value)
    if value == "":
        return ""
    if value_type in {"long", "int"}:
        try:
            return int(value)
        except ValueError:
            return value
    if value_type in {"double", "float"}:
        try:
            return float(value)
        except ValueError:
            return value
    return value


def split_sep(value):
    if not value:
        return []
    return [part.strip() for part in str(value).split(SEP) if part.strip()]


def scoped_id(subject, raw_id):
    return f"{subject}:{raw_id}"


def stable_id(*parts):
    text = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def preview_text(text, length=180):
    if not text:
        return ""
    one_line = " ".join(str(text).split())
    return one_line[:length]


def sanitize_relationship_type(keywords):
    raw = ""
    if keywords:
        raw = str(keywords).split(",")[0].strip()
    raw = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").upper()
    if not raw:
        raw = "RELATED_TO"
    if raw[0].isdigit():
        raw = f"REL_{raw}"
    return raw[:80]


def parse_graphml(graphml_path, subject, single_relationship_type=False):
    tree = ET.parse(graphml_path)
    root = tree.getroot()

    key_meta = {}
    for key in root.findall("g:key", GRAPHML_NS):
        key_id = key.attrib.get("id")
        key_meta[key_id] = {
            "name": key.attrib.get("attr.name", key_id),
            "type": key.attrib.get("attr.type", "string"),
        }

    def data_for(element):
        values = {}
        for data in element.findall("g:data", GRAPHML_NS):
            key_id = data.attrib.get("key")
            meta = key_meta.get(key_id, {"name": key_id, "type": "string"})
            values[meta["name"]] = coerce_graphml_value(data.text, meta["type"])
        return values

    nodes = []
    for node in root.findall(".//g:node", GRAPHML_NS):
        graphml_id = clean_value(node.attrib.get("id"))
        values = data_for(node)
        name = clean_value(values.get("entity_id") or graphml_id)
        source_ids = split_sep(values.get("source_id"))
        file_paths = split_sep(values.get("file_path"))

        nodes.append(
            {
                "id": scoped_id(subject, name),
                "name": name,
                "subject": subject,
                "entity_type": values.get("entity_type", ""),
                "description": values.get("description", ""),
                "source_id": values.get("source_id", ""),
                "source_ids": source_ids,
                "file_path": values.get("file_path", ""),
                "file_paths": file_paths,
                "created_at": values.get("created_at", ""),
                "truncate": values.get("truncate", ""),
                "displayName": name,
                "title": name,
            }
        )

    edges = []
    for edge in root.findall(".//g:edge", GRAPHML_NS):
        source_name = clean_value(edge.attrib.get("source"))
        target_name = clean_value(edge.attrib.get("target"))
        values = data_for(edge)
        source_ids = split_sep(values.get("source_id"))
        file_paths = split_sep(values.get("file_path"))
        rel_type = sanitize_relationship_type(values.get("keywords"))
        neo4j_rel_type = "RELATED_TO" if single_relationship_type else rel_type

        edges.append(
            {
                "id": stable_id(
                    subject,
                    source_name,
                    target_name,
                    rel_type,
                    values.get("keywords", ""),
                    values.get("source_id", ""),
                ),
                "source_entity_id": scoped_id(subject, source_name),
                "target_entity_id": scoped_id(subject, target_name),
                "source_name": source_name,
                "target_name": target_name,
                "subject": subject,
                "neo4j_relation_type": neo4j_rel_type,
                "relation_type": rel_type,
                "weight": values.get("weight", 0.0),
                "description": values.get("description", ""),
                "keywords": values.get("keywords", ""),
                "source_id": values.get("source_id", ""),
                "source_ids": source_ids,
                "file_path": values.get("file_path", ""),
                "file_paths": file_paths,
                "created_at": values.get("created_at", ""),
                "truncate": values.get("truncate", ""),
                "displayName": values.get("keywords", "") or rel_type,
                "title": values.get("keywords", "") or rel_type,
            }
        )

    return nodes, edges


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_documents(subject_dir, subject):
    data = load_json(subject_dir / "kv_store_full_docs.json")
    docs = []
    for doc_id, value in data.items():
        if not isinstance(value, dict):
            continue
        raw_id = value.get("_id") or doc_id
        content = value.get("content", "")
        docs.append(
            {
                "id": scoped_id(subject, raw_id),
                "doc_id": raw_id,
                "subject": subject,
                "content": content,
                "content_preview": preview_text(content),
                "file_path": value.get("file_path", ""),
                "create_time": value.get("create_time", ""),
                "update_time": value.get("update_time", ""),
                "displayName": raw_id,
                "title": raw_id,
            }
        )
    return docs


def build_chunks(subject_dir, subject):
    data = load_json(subject_dir / "kv_store_text_chunks.json")
    chunks = []
    for chunk_id, value in data.items():
        if not isinstance(value, dict):
            continue
        raw_id = value.get("_id") or chunk_id
        content = value.get("content", "")
        full_doc_id = value.get("full_doc_id", "")
        chunks.append(
            {
                "id": scoped_id(subject, raw_id),
                "chunk_id": raw_id,
                "subject": subject,
                "content": content,
                "content_preview": preview_text(content),
                "tokens": value.get("tokens", ""),
                "chunk_order_index": value.get("chunk_order_index", ""),
                "full_doc_id": full_doc_id,
                "full_doc_node_id": scoped_id(subject, full_doc_id)
                if full_doc_id
                else "",
                "file_path": value.get("file_path", ""),
                "create_time": value.get("create_time", ""),
                "update_time": value.get("update_time", ""),
                "displayName": raw_id,
                "title": raw_id,
            }
        )
    return chunks


def build_entity_chunk_links(entities):
    links = []
    for entity in entities:
        for chunk_id in entity.get("source_ids", []):
            links.append(
                {
                    "id": stable_id("entity_chunk", entity["id"], chunk_id),
                    "entity_id": entity["id"],
                    "chunk_id": scoped_id(entity["subject"], chunk_id),
                    "chunk_raw_id": chunk_id,
                    "subject": entity["subject"],
                }
            )
    return links


def build_document_chunk_links(chunks):
    links = []
    for chunk in chunks:
        if not chunk.get("full_doc_id"):
            continue
        links.append(
            {
                "id": stable_id(
                    "document_chunk",
                    chunk["full_doc_node_id"],
                    chunk["id"],
                ),
                "document_id": chunk["full_doc_node_id"],
                "chunk_id": chunk["id"],
                "subject": chunk["subject"],
                "order_index": chunk.get("chunk_order_index", ""),
            }
        )
    return links


def discover_subject_dirs(storage_dir, requested_subjects):
    if not storage_dir.exists():
        raise FileNotFoundError(f"Storage directory not found: {storage_dir}")

    available = {
        path.name: path
        for path in sorted(storage_dir.iterdir())
        if path.is_dir() and (path / GRAPHML_NAME).exists()
    }
    if requested_subjects:
        missing = [subject for subject in requested_subjects if subject not in available]
        if missing:
            raise FileNotFoundError(
                "Subjects not found or missing GraphML: " + ", ".join(missing)
            )
        return [available[subject] for subject in requested_subjects]
    return list(available.values())


def build_import_payload(
    subject_dirs,
    include_docs=True,
    include_chunks=True,
    single_relationship_type=False,
):
    subjects = []
    entities = []
    edges = []
    documents = []
    chunks = []

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        graphml_path = subject_dir / GRAPHML_NAME
        graph_entities, graph_edges = parse_graphml(
            graphml_path,
            subject,
            single_relationship_type=single_relationship_type,
        )

        subjects.append(
            {
                "id": subject,
                "name": subject,
                "storage_path": str(subject_dir),
                "graphml_path": str(graphml_path),
                "displayName": subject,
                "title": subject,
            }
        )
        entities.extend(graph_entities)
        edges.extend(graph_edges)

        if include_docs:
            documents.extend(build_documents(subject_dir, subject))
        if include_chunks:
            chunks.extend(build_chunks(subject_dir, subject))

    return {
        "subjects": subjects,
        "entities": entities,
        "edges": edges,
        "documents": documents,
        "chunks": chunks,
        "entity_chunk_links": build_entity_chunk_links(entities)
        if include_chunks
        else [],
        "document_chunk_links": build_document_chunk_links(chunks)
        if include_docs and include_chunks
        else [],
    }


def print_counts(payload):
    edge_type_counts = defaultdict(int)
    for edge in payload["edges"]:
        edge_type_counts[edge["neo4j_relation_type"]] += 1

    log("Prepared Neo4j import payload:")
    log(f"  subjects: {len(payload['subjects'])}")
    log(f"  entities: {len(payload['entities'])}")
    log(f"  entity relationships: {len(payload['edges'])}")
    log(f"  documents: {len(payload['documents'])}")
    log(f"  chunks: {len(payload['chunks'])}")
    log(f"  entity->chunk links: {len(payload['entity_chunk_links'])}")
    log(f"  document->chunk links: {len(payload['document_chunk_links'])}")
    log(f"  relationship types: {len(edge_type_counts)}")


def log(message=""):
    print(message, flush=True)


def render_progress(label, done, total):
    if total <= 0:
        log(f"{label}: skipped (0)")
        return

    width = 30
    done = min(done, total)
    filled = int(width * done / total)
    bar = "#" * filled + "." * (width - filled)
    percent = 100 * done / total
    end = "\n" if done >= total else ""
    print(
        f"\r{label:<34} [{bar}] {done}/{total} ({percent:5.1f}%)",
        end=end,
        flush=True,
    )


def query_count(session, query, **params):
    record = session.run(query, **params).single()
    if record is None:
        return 0
    return record[0]


def execute_batch(tx, query, param_name, rows):
    tx.run(query, {param_name: rows}).consume()


def run_in_batches(session, query, param_name, rows, batch_size, label):
    total = len(rows)
    if total == 0:
        render_progress(label, 0, 0)
        return

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        session.execute_write(execute_batch, query, param_name, batch)
        render_progress(label, start + len(batch), total)


def create_constraints(session):
    queries = [
        "CREATE CONSTRAINT subject_id IF NOT EXISTS FOR (n:Subject) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    ]
    log("Creating Neo4j constraints...")
    for index, query in enumerate(queries, start=1):
        session.run(query).consume()
        render_progress("Create constraints", index, len(queries))


def build_legacy_node_ids(payload):
    ids = set()
    ids.update(entity["name"] for entity in payload["entities"] if entity.get("name"))
    ids.update(doc["doc_id"] for doc in payload["documents"] if doc.get("doc_id"))
    ids.update(chunk["chunk_id"] for chunk in payload["chunks"] if chunk.get("chunk_id"))
    return sorted(ids)


def clear_imported_graph(session, payload, clear_all=False):
    if clear_all:
        total = query_count(session, "MATCH (n) RETURN count(n)")
        log(f"Clearing all Neo4j data: {total} nodes...")
        session.run("MATCH (n) DETACH DELETE n").consume()
        log("Cleared all Neo4j data.")
        return

    label_count = query_count(
        session,
        """
        MATCH (n)
        WHERE n:Subject OR n:Entity OR n:Document OR n:Chunk
        RETURN count(n)
        """,
    )
    log(f"Clearing imported-label nodes: {label_count} nodes...")
    session.run(
        """
        MATCH (n)
        WHERE n:Subject OR n:Entity OR n:Document OR n:Chunk
        DETACH DELETE n
        """
    ).consume()

    remaining = query_count(
        session,
        """
        MATCH (n)
        WHERE n:Subject OR n:Entity OR n:Document OR n:Chunk
        RETURN count(n)
        """,
    )
    log(f"Cleared imported-label nodes. Remaining: {remaining}.")

    legacy_ids = build_legacy_node_ids(payload)
    deleted_legacy = 0
    legacy_batch_size = 1000
    for start in range(0, len(legacy_ids), legacy_batch_size):
        id_batch = legacy_ids[start : start + legacy_batch_size]
        batch_count = query_count(
            session,
            """
            MATCH (n)
            WHERE n.id IN $ids OR n.name IN $ids OR n.displayName IN $ids
            RETURN count(n)
            """,
            ids=id_batch,
        )
        if batch_count:
            session.run(
                """
                MATCH (n)
                WHERE n.id IN $ids OR n.name IN $ids OR n.displayName IN $ids
                DETACH DELETE n
                """,
                ids=id_batch,
            ).consume()
            deleted_legacy += batch_count
        render_progress(
            "Clear legacy raw-name nodes",
            start + len(id_batch),
            len(legacy_ids),
        )
    log(f"Cleared legacy raw-name nodes: {deleted_legacy}.")


def import_subjects(session, subjects, batch_size):
    query = """
    UNWIND $subjects AS subject
    MERGE (s:Subject {id: subject.id})
    SET s += subject
    """
    run_in_batches(session, query, "subjects", subjects, batch_size, "Import subjects")


def import_entities(session, entities, batch_size):
    query = """
    UNWIND $entities AS entity
    MERGE (e:Entity {id: entity.id})
    SET e += entity
    WITH e, entity
    MATCH (s:Subject {id: entity.subject})
    MERGE (s)-[:HAS_ENTITY]->(e)
    """
    run_in_batches(session, query, "entities", entities, batch_size, "Import entities")


def import_documents(session, documents, batch_size):
    if not documents:
        render_progress("Import documents", 0, 0)
        return
    query = """
    UNWIND $documents AS document
    MERGE (d:Document {id: document.id})
    SET d += document
    WITH d, document
    MATCH (s:Subject {id: document.subject})
    MERGE (s)-[:HAS_DOCUMENT]->(d)
    """
    run_in_batches(
        session,
        query,
        "documents",
        documents,
        batch_size,
        "Import documents",
    )


def import_chunks(session, chunks, batch_size):
    if not chunks:
        render_progress("Import chunks", 0, 0)
        return
    query = """
    UNWIND $chunks AS chunk
    MERGE (c:Chunk {id: chunk.id})
    SET c += chunk
    WITH c, chunk
    MATCH (s:Subject {id: chunk.subject})
    MERGE (s)-[:HAS_CHUNK]->(c)
    """
    run_in_batches(session, query, "chunks", chunks, batch_size, "Import chunks")


def import_document_chunk_links(session, links, batch_size):
    if not links:
        render_progress("Link documents to chunks", 0, 0)
        return
    query = """
    UNWIND $links AS link
    MATCH (d:Document {id: link.document_id})
    MATCH (c:Chunk {id: link.chunk_id})
    MERGE (d)-[r:HAS_CHUNK {id: link.id}]->(c)
    SET r.subject = link.subject,
        r.order_index = link.order_index
    """
    run_in_batches(
        session,
        query,
        "links",
        links,
        batch_size,
        "Link documents to chunks",
    )


def import_entity_chunk_links(session, links, batch_size):
    if not links:
        render_progress("Link entities to chunks", 0, 0)
        return
    query = """
    UNWIND $links AS link
    MATCH (e:Entity {id: link.entity_id})
    MATCH (c:Chunk {id: link.chunk_id})
    MERGE (e)-[r:MENTIONED_IN {id: link.id}]->(c)
    SET r.subject = link.subject,
        r.chunk_raw_id = link.chunk_raw_id
    """
    run_in_batches(
        session,
        query,
        "links",
        links,
        batch_size,
        "Link entities to chunks",
    )


def apoc_available(session):
    try:
        session.run("RETURN apoc.version()").consume()
    except Neo4jError as exc:
        log(f"APOC is not available, using slower typed import fallback: {exc}")
        return False
    log("APOC is available, using batched dynamic relationship import.")
    return True


def import_entity_relationships_with_apoc(session, edges, batch_size):
    total = len(edges)
    if total == 0:
        render_progress("Import entity relationships", 0, 0)
        return

    query = """
    UNWIND $edges AS edge
    MATCH (source:Entity {id: edge.source_entity_id})
    MATCH (target:Entity {id: edge.target_entity_id})
    CALL apoc.merge.relationship(
        source,
        edge.neo4j_relation_type,
        {id: edge.id},
        edge,
        target,
        edge
    ) YIELD rel
    RETURN count(rel)
    """

    imported = 0
    for start in range(0, total, batch_size):
        batch = edges[start : start + batch_size]
        session.execute_write(execute_batch, query, "edges", batch)
        imported += len(batch)
        render_progress("Import entity relationships", imported, total)


def import_entity_relationships_by_type(session, edges, batch_size):
    grouped_edges = defaultdict(list)
    for edge in edges:
        grouped_edges[edge["neo4j_relation_type"]].append(edge)

    total = len(edges)
    if total == 0:
        render_progress("Import entity relationships", 0, 0)
        return

    imported = 0
    log(f"Importing entity relationships by {len(grouped_edges)} relationship types.")
    for rel_type, rel_edges in sorted(grouped_edges.items()):
        query = f"""
        UNWIND $edges AS edge
        MATCH (source:Entity {{id: edge.source_entity_id}})
        MATCH (target:Entity {{id: edge.target_entity_id}})
        MERGE (source)-[r:`{rel_type}` {{id: edge.id}}]->(target)
        SET r += edge
        """
        for start in range(0, len(rel_edges), batch_size):
            batch = rel_edges[start : start + batch_size]
            session.execute_write(execute_batch, query, "edges", batch)
            imported += len(batch)
            render_progress("Import entity relationships", imported, total)


def import_entity_relationships(session, edges, batch_size, use_apoc=True):
    if use_apoc and apoc_available(session):
        import_entity_relationships_with_apoc(session, edges, batch_size)
        return
    import_entity_relationships_by_type(session, edges, batch_size)


def print_database_summary(session):
    queries = [
        ("subjects", "MATCH (:Subject) RETURN count(*)"),
        ("entities", "MATCH (:Entity) RETURN count(*)"),
        ("documents", "MATCH (:Document) RETURN count(*)"),
        ("chunks", "MATCH (:Chunk) RETURN count(*)"),
        ("subject->entity links", "MATCH (:Subject)-[:HAS_ENTITY]->(:Entity) RETURN count(*)"),
        ("subject->document links", "MATCH (:Subject)-[:HAS_DOCUMENT]->(:Document) RETURN count(*)"),
        ("subject->chunk links", "MATCH (:Subject)-[:HAS_CHUNK]->(:Chunk) RETURN count(*)"),
        ("document->chunk links", "MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*)"),
        ("entity->chunk links", "MATCH (:Entity)-[:MENTIONED_IN]->(:Chunk) RETURN count(*)"),
        ("entity relationships", "MATCH (:Entity)-[r]->(:Entity) RETURN count(r)"),
        ("entity relationship types", "MATCH (:Entity)-[r]->(:Entity) RETURN count(DISTINCT type(r))"),
    ]

    log("Neo4j counts after import:")
    for label, query in queries:
        count = query_count(session, query)
        log(f"  {label}: {count}")


def import_payload(payload, args):
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password = os.getenv("NEO4J_PASSWORD", os.getenv("NEO4J_PASS"))
    neo4j_database = os.getenv("NEO4J_DATABASE") or None

    if not neo4j_password:
        raise RuntimeError(
            "Missing Neo4j password. Set NEO4J_PASSWORD in .env or environment."
        )

    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_username, neo4j_password),
    )
    try:
        with driver.session(database=neo4j_database) as session:
            if args.overwrite or args.clear_all:
                clear_imported_graph(
                    session,
                    payload,
                    clear_all=args.clear_all,
                )

            create_constraints(session)

            import_subjects(session, payload["subjects"], args.node_batch_size)
            import_documents(session, payload["documents"], args.node_batch_size)
            import_chunks(session, payload["chunks"], args.node_batch_size)
            import_entities(session, payload["entities"], args.node_batch_size)
            import_document_chunk_links(
                session,
                payload["document_chunk_links"],
                args.edge_batch_size,
            )
            import_entity_chunk_links(
                session,
                payload["entity_chunk_links"],
                args.edge_batch_size,
            )
            import_entity_relationships(
                session,
                payload["edges"],
                args.edge_batch_size,
                use_apoc=not args.no_apoc,
            )
            print_database_summary(session)
    finally:
        driver.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import LightRAG storage GraphML/docs/chunks into Neo4j."
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path(os.getenv("RAG_STORAGE_DIR", DEFAULT_STORAGE_DIR)),
        help="Storage directory containing subject subdirectories.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional subject directory names to import, e.g. C_program cybersec_lab.",
    )
    overwrite_default = env_bool("NEO4J_OVERWRITE", True)
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Delete existing Subject/Entity/Document/Chunk graph before import.",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not delete existing imported graph before import.",
    )
    parser.set_defaults(overwrite=overwrite_default)
    parser.add_argument(
        "--clear-all",
        action="store_true",
        default=env_bool("NEO4J_CLEAR_ALL", False),
        help="Delete the entire Neo4j database before import.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Do not import kv_store_full_docs.json document nodes.",
    )
    parser.add_argument(
        "--skip-chunks",
        action="store_true",
        help="Do not import kv_store_text_chunks.json chunk nodes.",
    )
    parser.add_argument(
        "--single-relationship-type",
        action="store_true",
        help=(
            "Use one Neo4j relationship type, RELATED_TO. By default the script "
            "uses keyword-derived relationship types."
        ),
    )
    parser.add_argument(
        "--no-apoc",
        action="store_true",
        help=(
            "Disable APOC-based dynamic relationship import and use the slower "
            "pure Cypher grouped-by-type fallback."
        ),
    )
    parser.add_argument(
        "--node-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE_NODES,
    )
    parser.add_argument(
        "--edge-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE_EDGES,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse storage and print counts without connecting to Neo4j.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    storage_dir = args.storage_dir.expanduser().resolve()
    subject_dirs = discover_subject_dirs(storage_dir, args.subjects)

    if not subject_dirs:
        raise RuntimeError(f"No subject GraphML files found under {storage_dir}")

    log(
        "Importing subjects: "
        + ", ".join(subject_dir.name for subject_dir in subject_dirs)
    )
    payload = build_import_payload(
        subject_dirs,
        include_docs=not args.skip_docs,
        include_chunks=not args.skip_chunks,
        single_relationship_type=args.single_relationship_type,
    )
    print_counts(payload)

    if args.dry_run:
        log("Dry run finished. Neo4j was not modified.")
        return

    import_payload(payload, args)
    log("Neo4j import finished.")


if __name__ == "__main__":
    main()
