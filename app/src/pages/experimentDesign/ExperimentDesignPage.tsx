/**
 * Experiment Design page — Sprint 4.
 *
 * Provides a UI for creating, viewing, and managing factorial experiment designs.
 * Uses REST API endpoints under /v1/experiment-designs.
 */
import { css } from "@emotion/react";
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import type { ColumnDef } from "@tanstack/react-table";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";

import {
  Button,
  Flex,
  Heading,
  Icon,
  Icons,
  Loading,
  Text,
  View,
} from "@phoenix/components";
import { tableCSS } from "@phoenix/components/table/styles";
import { TimestampCell } from "@phoenix/components/table/TimestampCell";

/* ── Shared styles ──────────────────────────────────────────────────────── */

const tabBarCSS = css`
  display: flex;
  gap: var(--ac-global-dimension-size-200);
  border-bottom: 1px solid var(--ac-global-color-grey-300);
  padding: 0 var(--ac-global-dimension-size-200);
`;

const tabCSS = css`
  padding: var(--ac-global-dimension-size-100)
    var(--ac-global-dimension-size-200);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  color: var(--ac-global-text-color-700);
  background: none;
  border-top: none;
  border-left: none;
  border-right: none;
  font-size: var(--ac-global-dimension-font-size-100);
  &:hover {
    color: var(--ac-global-text-color-900);
  }
`;

const activeTabCSS = css`
  ${tabCSS};
  border-bottom-color: var(--ac-global-color-primary);
  color: var(--ac-global-text-color-900);
  font-weight: 600;
`;

const statusBadgeCSS = css`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 500;
`;

const cardCSS = css`
  border: 1px solid var(--ac-global-color-grey-300);
  border-radius: var(--ac-global-rounding-medium);
  padding: var(--ac-global-dimension-size-200);
  cursor: pointer;
  transition: border-color 0.15s;
  &:hover {
    border-color: var(--ac-global-color-primary);
  }
`;

const formFieldCSS = css`
  display: flex;
  flex-direction: column;
  gap: var(--ac-global-dimension-size-50);
  label {
    font-size: 13px;
    font-weight: 600;
    color: var(--ac-global-text-color-700);
  }
  input,
  select,
  textarea {
    padding: 6px 10px;
    border: 1px solid var(--ac-global-color-grey-300);
    border-radius: var(--ac-global-rounding-small);
    font-size: 14px;
    background: var(--ac-global-background-color-light);
    color: var(--ac-global-text-color-900);
  }
`;

function StatusBadge({ status }: { status: string }) {
  const colorMap: Record<string, string> = {
    draft: "var(--ac-global-color-grey-500)",
    cells_generated: "var(--ac-global-color-primary)",
    running: "var(--ac-global-color-primary)",
    completed: "var(--ac-global-color-green-700)",
    failed: "var(--ac-global-color-danger)",
  };
  const color = colorMap[status] || colorMap.draft;
  return (
    <span
      css={css`
        ${statusBadgeCSS};
        color: ${color};
        border: 1px solid ${color};
      `}
    >
      {status.replace("_", " ")}
    </span>
  );
}

/* ── Types ──────────────────────────────────────────────────────────────── */

interface Factor {
  id: number;
  designId: number;
  name: string;
  factorType: string;
  levels: unknown[];
  createdAt: string;
}

interface DesignCell {
  id: number;
  designId: number;
  combination: Record<string, unknown>;
  status: string;
  experimentId: number | null;
  resultSummary: Record<string, unknown>;
  createdAt: string;
}

interface Design {
  id: number;
  name: string;
  description: string | null;
  designType: string;
  status: string;
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  factors: Factor[];
  cells: DesignCell[];
}

interface Template {
  id: string;
  name: string;
  description: string;
}

/* ── REST helpers (camelCase ↔ snake_case) ──────────────────────────────── */

function snakeToCamel(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const camelKey = key.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
    if (Array.isArray(value)) {
      out[camelKey] = value.map((v) =>
        typeof v === "object" && v !== null
          ? snakeToCamel(v as Record<string, unknown>)
          : v
      );
    } else if (typeof value === "object" && value !== null) {
      out[camelKey] = snakeToCamel(value as Record<string, unknown>);
    } else {
      out[camelKey] = value;
    }
  }
  return out;
}

async function fetchDesigns(): Promise<Design[]> {
  const res = await fetch("/v1/experiment-designs");
  const json = await res.json();
  return (json.data ?? []).map(
    (d: Record<string, unknown>) => snakeToCamel(d) as unknown as Design
  );
}

async function fetchTemplates(): Promise<Template[]> {
  const res = await fetch("/v1/experiment-designs/templates");
  const json = await res.json();
  return json.data ?? [];
}

async function createDesignFromTemplate(templateName: string): Promise<Design> {
  const res = await fetch("/v1/experiment-designs/from-template", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ template_name: templateName }),
  });
  const json = await res.json();
  return snakeToCamel(json.data) as unknown as Design;
}

async function createDesign(body: {
  name: string;
  description?: string;
  design_type?: string;
}): Promise<Design> {
  const res = await fetch("/v1/experiment-designs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = await res.json();
  return snakeToCamel(json.data) as unknown as Design;
}

async function generateCells(designId: number): Promise<DesignCell[]> {
  const res = await fetch(
    `/v1/experiment-designs/${designId}/generate-cells`,
    { method: "POST" }
  );
  const json = await res.json();
  return (json.data ?? []).map(
    (c: Record<string, unknown>) => snakeToCamel(c) as unknown as DesignCell
  );
}

async function runDesign(designId: number): Promise<Design> {
  const res = await fetch(`/v1/experiment-designs/${designId}/run`, {
    method: "POST",
  });
  const json = await res.json();
  return snakeToCamel(json.data) as unknown as Design;
}

async function deleteDesign(designId: number): Promise<void> {
  await fetch(`/v1/experiment-designs/${designId}`, { method: "DELETE" });
}

/* ── Column definitions ─────────────────────────────────────────────────── */

const designColumns: ColumnDef<Design>[] = [
  { header: "Name", accessorKey: "name", size: 200 },
  {
    header: "Type",
    accessorKey: "designType",
    size: 140,
    cell: ({ getValue }) => (
      <Text>{(getValue() as string).replace("_", " ")}</Text>
    ),
  },
  {
    header: "Status",
    accessorKey: "status",
    size: 130,
    cell: ({ getValue }) => <StatusBadge status={getValue() as string} />,
  },
  {
    header: "Factors",
    accessorKey: "factors",
    size: 80,
    cell: ({ getValue }) => <Text>{(getValue() as Factor[]).length}</Text>,
  },
  {
    header: "Cells",
    accessorKey: "cells",
    size: 80,
    cell: ({ getValue }) => <Text>{(getValue() as DesignCell[]).length}</Text>,
  },
  {
    header: "Created",
    accessorKey: "createdAt",
    size: 180,
    cell: ({ getValue }) => <TimestampCell datetime={getValue() as string} />,
  },
];

/* ── Page component ─────────────────────────────────────────────────────── */

export function ExperimentDesignPage() {
  return (
    <Suspense fallback={<Loading />}>
      <ExperimentDesignContent />
    </Suspense>
  );
}

function ExperimentDesignContent() {
  const [activeTab, setActiveTab] = useState<
    "designs" | "templates" | "detail"
  >("designs");
  const [designs, setDesigns] = useState<Design[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [selectedDesign, setSelectedDesign] = useState<Design | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [d, t] = await Promise.all([fetchDesigns(), fetchTemplates()]);
      setDesigns(d);
      setTemplates(t);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleCreateFromTemplate = useCallback(
    async (templateName: string) => {
      const design = await createDesignFromTemplate(templateName);
      setSelectedDesign(design);
      setActiveTab("detail");
      void refresh();
    },
    [refresh]
  );

  const handleCreateBlank = useCallback(async () => {
    const design = await createDesign({ name: "New Experiment Design" });
    setSelectedDesign(design);
    setActiveTab("detail");
    void refresh();
  }, [refresh]);

  const handleDeleteDesign = useCallback(
    async (designId: number) => {
      await deleteDesign(designId);
      setSelectedDesign(null);
      setActiveTab("designs");
      void refresh();
    },
    [refresh]
  );

  const handleGenerateCells = useCallback(
    async (designId: number) => {
      await generateCells(designId);
      void refresh();
      // Refresh the selected design
      const d = await Promise.all([fetchDesigns()]);
      const updated = d[0].find((x) => x.id === designId);
      if (updated) setSelectedDesign(updated);
    },
    [refresh]
  );

  const handleRunDesign = useCallback(
    async (designId: number) => {
      const updated = await runDesign(designId);
      setSelectedDesign(updated);
      void refresh();
    },
    [refresh]
  );

  if (loading) return <Loading />;

  return (
    <Flex direction="column" height="100%">
      {/* Header */}
      <View
        padding="size-200"
        flex="none"
        borderBottomWidth="thin"
        borderBottomColor="dark"
      >
        <Flex
          direction="row"
          justifyContent="space-between"
          alignItems="center"
        >
          <Flex direction="row" alignItems="center" gap="size-100">
            <Icon svg={<Icons.CubeOutline />} />
            <Heading level={1}>Experiment Design</Heading>
          </Flex>
          <Flex direction="row" gap="size-100">
            <Button variant="default" size="S" onClick={() => void refresh()}>
              Refresh
            </Button>
            <Button
              variant="primary"
              size="S"
              onClick={() => void handleCreateBlank()}
            >
              New Design
            </Button>
          </Flex>
        </Flex>
      </View>

      {/* Tabs */}
      <div css={tabBarCSS}>
        <button
          css={activeTab === "designs" ? activeTabCSS : tabCSS}
          onClick={() => {
            setActiveTab("designs");
            setSelectedDesign(null);
          }}
        >
          Designs ({designs.length})
        </button>
        <button
          css={activeTab === "templates" ? activeTabCSS : tabCSS}
          onClick={() => setActiveTab("templates")}
        >
          Templates ({templates.length})
        </button>
        {selectedDesign && (
          <button
            css={activeTab === "detail" ? activeTabCSS : tabCSS}
            onClick={() => setActiveTab("detail")}
          >
            {selectedDesign.name}
          </button>
        )}
      </div>

      {/* Content */}
      <View flex="1 1 auto" overflow="auto" padding="size-200">
        {activeTab === "designs" && (
          <DesignList
            designs={designs}
            onSelect={(d) => {
              setSelectedDesign(d);
              setActiveTab("detail");
            }}
          />
        )}
        {activeTab === "templates" && (
          <TemplateSelector
            templates={templates}
            onSelect={handleCreateFromTemplate}
          />
        )}
        {activeTab === "detail" && selectedDesign && (
          <DesignDetail
            design={selectedDesign}
            onDelete={handleDeleteDesign}
            onGenerateCells={handleGenerateCells}
            onRun={handleRunDesign}
          />
        )}
      </View>
    </Flex>
  );
}

/* ── Design list ────────────────────────────────────────────────────────── */

function DesignList({
  designs,
  onSelect,
}: {
  designs: Design[];
  onSelect: (d: Design) => void;
}) {
  if (designs.length === 0) {
    return (
      <EmptyState
        title="No experiment designs"
        description="Create a design from a template or start from scratch."
      />
    );
  }

  return (
    <DataTable
      columns={designColumns}
      data={designs}
      onRowClick={(row) => onSelect(row)}
    />
  );
}

/* ── Template selector ──────────────────────────────────────────────────── */

function TemplateSelector({
  templates,
  onSelect,
}: {
  templates: Template[];
  onSelect: (templateName: string) => void;
}) {
  if (templates.length === 0) {
    return (
      <EmptyState
        title="No templates available"
        description="Templates will be loaded from the server."
      />
    );
  }

  return (
    <Flex direction="column" gap="size-200">
      <Text>
        Select a pre-built template to quickly configure an experiment design:
      </Text>
      <div
        css={css`
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: var(--ac-global-dimension-size-200);
        `}
      >
        {templates.map((t) => (
          <div
            key={t.id}
            css={cardCSS}
            role="button"
            tabIndex={0}
            onClick={() => onSelect(t.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onSelect(t.id);
            }}
          >
            <Flex direction="column" gap="size-50">
              <Text weight="heavy">{t.name}</Text>
              <Text color="text-500">{t.description}</Text>
            </Flex>
          </div>
        ))}
      </div>
    </Flex>
  );
}

/* ── Design detail (factors + matrix) ───────────────────────────────────── */

function DesignDetail({
  design,
  onDelete,
  onGenerateCells,
  onRun,
}: {
  design: Design;
  onDelete: (id: number) => void;
  onGenerateCells: (id: number) => void;
  onRun: (id: number) => void;
}) {
  return (
    <Flex direction="column" gap="size-200">
      {/* Design info */}
      <View
        padding="size-200"
        borderWidth="thin"
        borderColor="dark"
        borderRadius="medium"
      >
        <Flex direction="row" justifyContent="space-between" alignItems="start">
          <Flex direction="column" gap="size-50">
            <Heading level={2}>{design.name}</Heading>
            {design.description && (
              <Text color="text-500">{design.description}</Text>
            )}
            <Flex direction="row" gap="size-200">
              <Text>
                Type: <strong>{design.designType.replace("_", " ")}</strong>
              </Text>
              <StatusBadge status={design.status} />
            </Flex>
          </Flex>
          <Flex direction="row" gap="size-100">
            {design.status === "draft" && design.factors.length > 0 && (
              <Button
                variant="primary"
                size="S"
                onClick={() => onGenerateCells(design.id)}
              >
                Generate Cells
              </Button>
            )}
            {(design.status === "cells_generated" ||
              design.status === "failed") && (
              <Button
                variant="primary"
                size="S"
                onClick={() => onRun(design.id)}
              >
                Run All
              </Button>
            )}
            <Button
              variant="danger"
              size="S"
              onClick={() => onDelete(design.id)}
            >
              Delete
            </Button>
          </Flex>
        </Flex>
      </View>

      {/* Factors */}
      <View>
        <Heading level={3}>Factors ({design.factors.length})</Heading>
        {design.factors.length === 0 ? (
          <Text color="text-500">
            No factors defined. Add factors via the API.
          </Text>
        ) : (
          <Flex direction="column" gap="size-100">
            {design.factors.map((f) => (
              <View
                key={f.id}
                padding="size-100"
                borderWidth="thin"
                borderColor="dark"
                borderRadius="small"
              >
                <Flex direction="row" gap="size-200" alignItems="center">
                  <Text weight="heavy">{f.name}</Text>
                  <StatusBadge status={f.factorType} />
                  <Text color="text-500">
                    {(f.levels ?? []).length} level
                    {(f.levels ?? []).length !== 1 ? "s" : ""}
                  </Text>
                  <Text color="text-500" size="XS">
                    {(f.levels ?? [])
                      .map((l: unknown) => {
                        if (typeof l === "object" && l !== null && "name" in l)
                          return (l as { name: string }).name;
                        return JSON.stringify(l);
                      })
                      .join(", ")}
                  </Text>
                </Flex>
              </View>
            ))}
          </Flex>
        )}
      </View>

      {/* Cell matrix */}
      <ExperimentMatrix cells={design.cells} factors={design.factors} />
    </Flex>
  );
}

/* ── Experiment matrix ──────────────────────────────────────────────────── */

function ExperimentMatrix({
  cells,
  factors,
}: {
  cells: DesignCell[];
  factors: Factor[];
}) {
  const matrixColumns = useMemo<ColumnDef<DesignCell>[]>(() => {
    const cols: ColumnDef<DesignCell>[] = factors.map((f) => ({
      header: f.name,
      accessorFn: (row: DesignCell) => {
        const val = row.combination[f.name];
        if (typeof val === "object" && val !== null && "name" in val) {
          return (val as { name: string }).name;
        }
        return JSON.stringify(val);
      },
      size: 160,
    }));
    cols.push({
      header: "Status",
      accessorKey: "status",
      size: 120,
      cell: ({ getValue }) => <StatusBadge status={getValue() as string} />,
    });
    return cols;
  }, [factors]);

  if (cells.length === 0) {
    return (
      <View>
        <Heading level={3}>Cell Matrix</Heading>
        <Text color="text-500">
          No cells generated yet. Add factors and generate cells.
        </Text>
      </View>
    );
  }

  return (
    <View>
      <Heading level={3}>
        Cell Matrix ({cells.length} cell{cells.length !== 1 ? "s" : ""})
      </Heading>
      <SimpleTable columns={matrixColumns} data={cells} />
    </View>
  );
}

/* ── Reusable table ─────────────────────────────────────────────────────── */

function DataTable<T>({
  columns,
  data,
  onRowClick,
}: {
  columns: ColumnDef<T>[];
  data: T[];
  onRowClick?: (row: T) => void;
}) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <table css={tableCSS}>
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id}>
            {hg.headers.map((h) => (
              <th key={h.id} style={{ width: h.getSize() }}>
                {h.isPlaceholder
                  ? null
                  : flexRender(h.column.columnDef.header, h.getContext())}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr
            key={row.id}
            onClick={() => onRowClick?.(row.original)}
            css={
              onRowClick
                ? css`
                    cursor: pointer;
                    &:hover {
                      background: var(--ac-global-color-grey-100);
                    }
                  `
                : undefined
            }
          >
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SimpleTable<T>({
  columns,
  data,
}: {
  columns: ColumnDef<T>[];
  data: T[];
}) {
  return <DataTable columns={columns} data={data} />;
}

/* ── Empty state ────────────────────────────────────────────────────────── */

function EmptyState({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <Flex
      direction="column"
      alignItems="center"
      justifyContent="center"
      height="100%"
      gap="size-100"
    >
      <Icon
        svg={<Icons.CubeOutline />}
        color="var(--ac-global-text-color-500)"
      />
      <Heading level={3}>{title}</Heading>
      <Text color="text-500">{description}</Text>
    </Flex>
  );
}
