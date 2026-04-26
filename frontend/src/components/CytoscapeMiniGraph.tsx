import cytoscape, { Core } from "cytoscape";
import { useEffect, useRef } from "react";
import type { GraphEdge, GraphNode } from "../types";

interface CytoscapeMiniGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  centerEntityIds: string[];
  onSelect: (node: GraphNode | null) => void;
}

export function CytoscapeMiniGraph({ nodes, edges, centerEntityIds, onSelect }: CytoscapeMiniGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const nodeMap = new Map(nodes.map((node) => [node.id, node]));
    cyRef.current?.destroy();
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: [
        ...nodes.map((node) => ({
          data: {
            id: node.id,
            label: node.label,
            hitType: centerEntityIds.includes(node.id) ? "direct" : node.hitType,
            type: node.type
          }
        })),
        ...edges.map((edge) => ({
          data: {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.label
          }
        }))
      ],
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-size": "9px",
            "text-wrap": "wrap",
            "text-max-width": "82px",
            color: "#292524",
            "background-color": "#a8a29e",
            "border-width": 1,
            "border-color": "#78716c",
            width: 30,
            height: 30
          }
        },
        {
          selector: 'node[hitType = "direct"]',
          style: {
            "background-color": "#f97316",
            "border-color": "#9a3412",
            width: 44,
            height: 44,
            "font-weight": 700
          }
        },
        {
          selector: 'node[hitType = "related"]',
          style: {
            "background-color": "#14b8a6",
            "border-color": "#0f766e",
            width: 36,
            height: 36
          }
        },
        {
          selector: "edge",
          style: {
            width: 1.2,
            "line-color": "#a8a29e",
            "target-arrow-color": "#a8a29e",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            opacity: 0.75
          }
        }
      ],
      layout: { name: "cose", animate: false, fit: true, padding: 22 },
      wheelSensitivity: 0.22
    });
    cyRef.current.on("tap", "node", (event) => onSelect(nodeMap.get(event.target.id()) || null));
    cyRef.current.on("tap", (event) => {
      if (event.target === cyRef.current) onSelect(null);
    });
    return () => cyRef.current?.destroy();
  }, [centerEntityIds, edges, nodes, onSelect]);

  return <div ref={containerRef} className="mini-graph" />;
}
