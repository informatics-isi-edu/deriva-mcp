// Dagre-based layout for ERD graph

import dagre from "dagre";
import type { Node, Edge } from "@xyflow/react";

const TABLE_NODE_WIDTH = 260;
const TABLE_NODE_HEIGHT = 72;
const SCHEMA_NODE_WIDTH = 240;
const SCHEMA_NODE_HEIGHT = 120;

export function layoutGraph(
  nodes: Node[],
  edges: Edge[],
  direction: "TB" | "LR" = "LR"
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: direction,
    nodesep: 40,
    ranksep: 80,
    marginx: 40,
    marginy: 40,
  });

  nodes.forEach((node) => {
    const isSchema = node.type === "schemaNode";
    g.setNode(node.id, {
      width: isSchema ? SCHEMA_NODE_WIDTH : TABLE_NODE_WIDTH,
      height: isSchema ? SCHEMA_NODE_HEIGHT : TABLE_NODE_HEIGHT,
    });
  });

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  const layoutedNodes = nodes.map((node) => {
    const pos = g.node(node.id);
    const isSchema = node.type === "schemaNode";
    const w = isSchema ? SCHEMA_NODE_WIDTH : TABLE_NODE_WIDTH;
    const h = isSchema ? SCHEMA_NODE_HEIGHT : TABLE_NODE_HEIGHT;
    return {
      ...node,
      position: {
        x: pos.x - w / 2,
        y: pos.y - h / 2,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
}
