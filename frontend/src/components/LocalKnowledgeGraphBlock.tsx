import { useState } from "react";
import type { GraphNode, LocalSubgraph } from "../types";
import { CytoscapeMiniGraph } from "./CytoscapeMiniGraph";

export function LocalKnowledgeGraphBlock({
  subgraphs,
  graphError
}: {
  subgraphs: LocalSubgraph[];
  graphError?: string;
}) {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  return (
    <div className="trace-block">
      <div className="trace-heading">知识图谱命中</div>
      {graphError ? <div className="empty-inline">{graphError}</div> : null}
      {!graphError && !subgraphs.length ? <div className="empty-inline">本次未命中可展示局部图谱。</div> : null}
      {subgraphs.map((subgraph) => (
        <section className="subgraph-card" key={subgraph.id}>
          <div className="subgraph-card-head">
            <div>
              <div className="font-semibold text-stone-950">{subgraph.title}</div>
              <div className="mt-1 text-xs text-stone-500">{subgraph.summary}</div>
            </div>
            <div className="subgraph-stat">{subgraph.nodes.length} nodes</div>
          </div>
          <CytoscapeMiniGraph
            nodes={subgraph.nodes}
            edges={subgraph.edges}
            centerEntityIds={subgraph.centerEntityIds}
            onSelect={setSelectedNode}
          />
        </section>
      ))}
      {selectedNode ? (
        <div className="node-detail-inline">
          <div className="font-semibold text-stone-950">{selectedNode.label}</div>
          <div className="mt-1 text-xs uppercase tracking-[0.14em] text-stone-500">
            {selectedNode.subjectId} · {selectedNode.type}
          </div>
          {selectedNode.metadata?.description ? (
            <p className="mt-2 text-sm leading-6 text-stone-600">
              {String(selectedNode.metadata.description)}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
