import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Node,
  type Edge,
  type NodeMouseHandler,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import TableNode from "./TableNode";
import SchemaNode from "./SchemaNode";
import Legend from "./Legend";
import { layoutGraph } from "@/layout";
import type { EnrichedTable, SchemaFilter, CatalogSchema } from "@/types";

const nodeTypes = { tableNode: TableNode, schemaNode: SchemaNode };

export interface CanvasControls {
  zoomIn: () => void;
  zoomOut: () => void;
  fitView: () => void;
  showMiniMap: boolean;
  setShowMiniMap: (v: boolean) => void;
  mapSize: "S" | "M" | "L";
  cycleMapSize: () => void;
}

interface ERDCanvasProps {
  tables: EnrichedTable[];
  schema: CatalogSchema;
  filter: SchemaFilter;
  hideAssociations: boolean;
  searchQuery: string;
  selectedTable: string | null;
  onSelectTable: (qualifiedName: string | null) => void;
  viewMode: "schemas" | "tables";
  activeSchema: string | null;
  onDrillIntoSchema: (schemaName: string) => void;
  onControlsReady?: (controls: CanvasControls) => void;
}

// ── Schema-level view ──────────────────────────────────────────────

function buildSchemaNodes(
  tables: EnrichedTable[],
  schema: CatalogSchema,
  onDrillIntoSchema: (schemaName: string) => void
): { nodes: Node[]; edges: Edge[] } {
  const bySchema: Record<string, EnrichedTable[]> = {};
  for (const t of tables) {
    (bySchema[t.schema] ??= []).push(t);
  }

  const nodes: Node[] = Object.entries(bySchema).map(([schemaName, schemaTables]) => ({
    id: schemaName,
    type: "schemaNode",
    position: { x: 0, y: 0 },
    data: {
      label: schemaName,
      tableCount: schemaTables.length,
      domainCount: schemaTables.filter((t) => t.tableType === "domain").length,
      vocabCount: schemaTables.filter((t) => t.tableType === "vocabulary").length,
      assetCount: schemaTables.filter((t) => t.tableType === "asset").length,
      assocCount: schemaTables.filter((t) => t.tableType === "association").length,
      isSelected: false,
      isMlSchema: schemaName === schema.ml_schema,
      comment: schema.schemas[schemaName]?.comment || "",
      onDrillIn: () => onDrillIntoSchema(schemaName),
    },
  }));

  const edges: Edge[] = [];
  const edgeSet = new Set<string>();
  for (const t of tables) {
    for (const fk of t.info.foreign_keys) {
      const refSchema = fk.referenced_table.split(".")[0];
      if (refSchema && refSchema !== t.schema && bySchema[refSchema]) {
        const edgeId = `${t.schema}->${refSchema}`;
        if (!edgeSet.has(edgeId)) {
          edgeSet.add(edgeId);
          edges.push({
            id: edgeId,
            source: t.schema,
            target: refSchema,
            style: { stroke: "#94a3b8", strokeWidth: 1.5 },
          });
        }
      }
    }
  }

  return layoutGraph(nodes, edges);
}

// ── Table-level view ───────────────────────────────────────────────

function buildTableNodes(
  tables: EnrichedTable[],
  activeSchema: string,
  filter: SchemaFilter,
  hideAssociations: boolean,
  searchQuery: string
): { nodes: Node[]; edges: Edge[] } {
  let filtered = tables.filter((t) => t.schema === activeSchema);

  if (filter !== "all") {
    filtered = filtered.filter((t) => t.tableType === filter);
  }
  if (hideAssociations) {
    filtered = filtered.filter((t) => !t.info.is_association);
  }
  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    filtered = filtered.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        (t.info.comment && t.info.comment.toLowerCase().includes(q))
    );
  }

  const visibleSet = new Set(filtered.map((t) => t.qualifiedName));

  const nodes: Node[] = filtered.map((t) => ({
    id: t.qualifiedName,
    type: "tableNode",
    position: { x: 0, y: 0 },
    data: {
      label: t.name,
      schema: t.schema,
      tableType: t.tableType,
      recordCount: t.recordCount,
      isSelected: false,
      comment: t.info.comment || "",
    },
  }));

  const edges: Edge[] = [];
  const edgeSet = new Set<string>();
  filtered.forEach((t) => {
    t.info.foreign_keys.forEach((fk) => {
      const target = fk.referenced_table;
      if (visibleSet.has(target) && target !== t.qualifiedName) {
        const edgeId = `${t.qualifiedName}->${target}`;
        if (!edgeSet.has(edgeId)) {
          edgeSet.add(edgeId);
          edges.push({
            id: edgeId,
            source: t.qualifiedName,
            target,
            label: fk.columns.join(", "),
            style: { stroke: "#94a3b8", strokeWidth: 1.5 },
            labelStyle: { fontSize: 10, fill: "#64748b", opacity: 0 },
            labelBgStyle: { fill: "white", opacity: 0 },
          });
        }
      }
    });
  });

  return layoutGraph(nodes, edges);
}

// Apply selection styling without re-layouting
function applySelection(
  nodes: Node[],
  edges: Edge[],
  selectedId: string | null
): { nodes: Node[]; edges: Edge[] } {
  const styledNodes = nodes.map((n) => ({
    ...n,
    data: { ...n.data, isSelected: n.id === selectedId },
  }));

  const styledEdges = edges.map((e) => {
    const isHighlighted =
      selectedId === e.source || selectedId === e.target;
    return {
      ...e,
      style: {
        stroke: isHighlighted ? "#334155" : "#94a3b8",
        strokeWidth: isHighlighted ? 2.5 : 1.5,
        opacity: selectedId && !isHighlighted ? 0.15 : 1,
      },
      labelStyle: {
        ...(e.labelStyle || {}),
        opacity: isHighlighted ? 1 : 0,
      },
      labelBgStyle: {
        ...(e.labelBgStyle || {}),
        opacity: isHighlighted ? 0.9 : 0,
      },
    };
  });

  return { nodes: styledNodes, edges: styledEdges };
}

// ── Main component ─────────────────────────────────────────────────

function ERDCanvasInner({
  tables,
  schema,
  filter,
  hideAssociations,
  searchQuery,
  selectedTable,
  onSelectTable,
  viewMode,
  activeSchema,
  onDrillIntoSchema,
  onControlsReady,
}: ERDCanvasProps) {
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  const [showMiniMap, setShowMiniMap] = useState(true);
  const [mapSize, setMapSize] = useState<"S" | "M" | "L">("M");

  const mapDimensions = { S: { w: 140, h: 100 }, M: { w: 200, h: 150 }, L: { w: 300, h: 220 } };
  const { w: mapW, h: mapH } = mapDimensions[mapSize];

  const cycleMapSize = useCallback(() => setMapSize((s) => (s === "S" ? "M" : s === "M" ? "L" : "S")), []);

  // Expose controls to parent
  useEffect(() => {
    onControlsReady?.({
      zoomIn: () => zoomIn({ duration: 200 }),
      zoomOut: () => zoomOut({ duration: 200 }),
      fitView: () => fitView({ padding: 0.2, duration: 300 }),
      showMiniMap,
      setShowMiniMap,
      mapSize,
      cycleMapSize,
    });
  }, [onControlsReady, zoomIn, zoomOut, fitView, showMiniMap, mapSize, cycleMapSize]);

  // Compute layout only when structure changes (not on selection)
  const { nodes: baseNodes, edges: baseEdges } = useMemo(() => {
    if (viewMode === "schemas") {
      return buildSchemaNodes(tables, schema, onDrillIntoSchema);
    } else {
      return buildTableNodes(
        tables,
        activeSchema!,
        filter,
        hideAssociations,
        searchQuery
      );
    }
  }, [tables, schema, filter, hideAssociations, searchQuery, viewMode, activeSchema, onDrillIntoSchema]);

  // Apply selection styling on top of layout
  const selectedId = viewMode === "schemas" ? activeSchema : selectedTable;
  const { nodes: styledNodes, edges: styledEdges } = useMemo(
    () => applySelection(baseNodes, baseEdges, selectedId),
    [baseNodes, baseEdges, selectedId]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(styledNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(styledEdges);

  // Track whether this is a structural change (needs fitView) vs just selection
  const prevStructureKey = useRef("");
  const structureKey = `${viewMode}:${activeSchema}:${filter}:${hideAssociations}:${searchQuery}`;

  useEffect(() => {
    setNodes(styledNodes);
    setEdges(styledEdges);

    if (structureKey !== prevStructureKey.current) {
      prevStructureKey.current = structureKey;
      // Delay fitView to allow React Flow to process the new nodes
      setTimeout(() => fitView({ padding: 0.2, duration: 300 }), 50);
    }
  }, [styledNodes, styledEdges, structureKey, setNodes, setEdges, fitView]);

  const onNodeClick: NodeMouseHandler = useCallback(
    (_, node) => {
      if (viewMode === "schemas") {
        onSelectTable(node.id === activeSchema ? null : node.id);
      } else {
        onSelectTable(node.id === selectedTable ? null : node.id);
      }
    },
    [viewMode, selectedTable, activeSchema, onSelectTable]
  );

  const onNodeDoubleClick: NodeMouseHandler = useCallback(
    (_, node) => {
      if (viewMode === "schemas") {
        onDrillIntoSchema(node.id);
      }
    },
    [viewMode, onDrillIntoSchema]
  );

  const onPaneClick = useCallback(() => {
    onSelectTable(null);
  }, [onSelectTable]);

  return (
    <div className="relative w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onNodeDoubleClick={onNodeDoubleClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1}
          color="#d1d5db"
        />
        {showMiniMap && (
          <MiniMap
            key={mapSize}
            pannable
            zoomable
            style={{ width: mapW, height: mapH }}
            nodeColor={(node: Node) => {
              if (node.type === "schemaNode") {
                return (node.data as any).isMlSchema ? "#b45309" : "#334155";
              }
              const data = node.data as any;
              switch (data.tableType) {
                case "domain": return "#334155";
                case "ml": return "#b45309";
                case "vocabulary": return "#047857";
                case "asset": return "#0369a1";
                case "association": return "#71717a";
                default: return "#94a3b8";
              }
            }}
            maskColor="rgba(255,255,255,0.7)"
            className="!bg-white !border-slate-300"
          />
        )}
      </ReactFlow>

      {viewMode === "tables" && <Legend />}
    </div>
  );
}

// Wrap with ReactFlow's internal provider awareness
export default function ERDCanvas(props: ERDCanvasProps) {
  return <ERDCanvasInner {...props} />;
}
