import { css } from "@emotion/react";
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import type { ColumnDef } from "@tanstack/react-table";
import { Suspense, useCallback, useMemo, useState } from "react";
import { graphql, useLazyLoadQuery } from "react-relay";

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

import type { DataGenerationPageQuery } from "./__generated__/DataGenerationPageQuery.graphql";
import { DataGenPipelineBuilder } from "./DataGenPipelineBuilder";

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

function StatusBadge({ status }: { status: string }) {
  const colorMap: Record<string, string> = {
    pending: "var(--ac-global-color-grey-500)",
    running: "var(--ac-global-color-primary)",
    completed: "var(--ac-global-color-green-700)",
    failed: "var(--ac-global-color-danger)",
    cancelled: "var(--ac-global-color-grey-600)",
  };
  const color = colorMap[status] || colorMap.pending;
  return (
    <span
      css={css`
        ${statusBadgeCSS};
        color: ${color};
        border: 1px solid ${color};
      `}
    >
      {status}
    </span>
  );
}

type JobRow = {
  id: string;
  name: string;
  status: string;
  corpusSource: string;
  createdAt: string;
  completedAt: string | null;
};

type AdapterRow = {
  id: string;
  name: string;
  provider: string;
  modelName: string;
  canGenerate: boolean;
  canJudge: boolean;
  createdAt: string;
};

const jobColumns: ColumnDef<JobRow>[] = [
  {
    header: "Name",
    accessorKey: "name",
    size: 200,
  },
  {
    header: "Status",
    accessorKey: "status",
    size: 120,
    cell: ({ getValue }) => <StatusBadge status={getValue() as string} />,
  },
  {
    header: "Corpus Source",
    accessorKey: "corpusSource",
    size: 200,
  },
  {
    header: "Created",
    accessorKey: "createdAt",
    size: 180,
    cell: ({ getValue }) => <TimestampCell datetime={getValue() as string} />,
  },
  {
    header: "Completed",
    accessorKey: "completedAt",
    size: 180,
    cell: ({ getValue }) => {
      const value = getValue() as string | null;
      return value ? <TimestampCell datetime={value} /> : <Text>—</Text>;
    },
  },
];

const adapterColumns: ColumnDef<AdapterRow>[] = [
  {
    header: "Name",
    accessorKey: "name",
    size: 180,
  },
  {
    header: "Provider",
    accessorKey: "provider",
    size: 140,
  },
  {
    header: "Model",
    accessorKey: "modelName",
    size: 180,
  },
  {
    header: "Generate",
    accessorKey: "canGenerate",
    size: 100,
    cell: ({ getValue }) => (getValue() ? "✓" : "—"),
  },
  {
    header: "Judge",
    accessorKey: "canJudge",
    size: 100,
    cell: ({ getValue }) => (getValue() ? "✓" : "—"),
  },
  {
    header: "Created",
    accessorKey: "createdAt",
    size: 180,
    cell: ({ getValue }) => <TimestampCell datetime={getValue() as string} />,
  },
];

export function DataGenerationPage() {
  return (
    <Suspense fallback={<Loading />}>
      <DataGenerationContent />
    </Suspense>
  );
}

const newJobPanelCSS = css`
  border: 1px solid var(--ac-global-color-grey-300);
  border-radius: var(--ac-global-rounding-medium);
  margin-bottom: var(--ac-global-dimension-size-200);
`;

const panelToggleCSS = css`
  display: flex;
  align-items: center;
  gap: var(--ac-global-dimension-size-100);
  padding: var(--ac-global-dimension-size-100)
    var(--ac-global-dimension-size-200);
  cursor: pointer;
  user-select: none;
  background: var(--ac-global-background-color-light);
  border: none;
  width: 100%;
  text-align: left;
  border-radius: var(--ac-global-rounding-medium);
  font-size: var(--ac-global-dimension-font-size-100);
  color: var(--ac-global-text-color-900);
  &:hover {
    background: var(--ac-global-color-grey-100);
  }
`;

function DataGenerationContent() {
  const [activeTab, setActiveTab] = useState<"jobs" | "adapters">("jobs");
  const [fetchKey, setFetchKey] = useState(0);
  const [showNewJob, setShowNewJob] = useState(false);

  const data = useLazyLoadQuery<DataGenerationPageQuery>(
    graphql`
      query DataGenerationPageQuery {
        dataGenerationJobs(first: 100) {
          edges {
            node {
              id
              name
              status
              corpusSourceType
              createdAt
              completedAt
            }
          }
        }
        llmAdapters(first: 100) {
          edges {
            node {
              id
              name
              provider
              modelName
              canGenerate
              canJudge
              createdAt
            }
          }
        }
      }
    `,
    {},
    {
      fetchKey,
      fetchPolicy: "store-and-network",
    }
  );

  const onRefresh = useCallback(() => {
    setFetchKey((prev) => prev + 1);
  }, []);

  const jobRows: JobRow[] = useMemo(
    () =>
      (data.dataGenerationJobs?.edges ?? []).map((edge) => ({
        id: edge.node.id,
        name: edge.node.name ?? "Untitled Job",
        status: edge.node.status,
        corpusSource: edge.node.corpusSourceType ?? "unknown",
        createdAt: edge.node.createdAt,
        completedAt: edge.node.completedAt ?? null,
      })),
    [data.dataGenerationJobs]
  );

  const adapterRows: AdapterRow[] = useMemo(
    () =>
      (data.llmAdapters?.edges ?? []).map((edge) => ({
        id: edge.node.id,
        name: edge.node.name,
        provider: edge.node.provider,
        modelName: edge.node.modelName,
        canGenerate: edge.node.canGenerate,
        canJudge: edge.node.canJudge,
        createdAt: edge.node.createdAt,
      })),
    [data.llmAdapters]
  );

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
            <Icon svg={<Icons.FileTextOutline />} />
            <Heading level={1}>Data Generation</Heading>
          </Flex>
          <Button variant="default" size="S" onClick={onRefresh}>
            Refresh
          </Button>
        </Flex>
      </View>

      {/* Tabs */}
      <div css={tabBarCSS}>
        <button
          css={activeTab === "jobs" ? activeTabCSS : tabCSS}
          onClick={() => setActiveTab("jobs")}
        >
          Jobs ({jobRows.length})
        </button>
        <button
          css={activeTab === "adapters" ? activeTabCSS : tabCSS}
          onClick={() => setActiveTab("adapters")}
        >
          LLM Adapters ({adapterRows.length})
        </button>
      </div>

      {/* Content */}
      <View flex="1 1 auto" overflow="auto" padding="size-200">
        {activeTab === "jobs" ? (
          <>
            {/* Collapsible New Job panel */}
            <div css={newJobPanelCSS}>
              <button
                css={panelToggleCSS}
                onClick={() => setShowNewJob((v) => !v)}
              >
                <Icon
                  svg={
                    showNewJob ? (
                      <Icons.ArrowIosDownwardOutline />
                    ) : (
                      <Icons.ArrowIosForwardOutline />
                    )
                  }
                />
                <Text weight="heavy">
                  {showNewJob ? "Hide New Job" : "New Job"}
                </Text>
              </button>
              {showNewJob && (
                <DataGenPipelineBuilder
                  onJobCreated={() => {
                    setFetchKey((prev) => prev + 1);
                    setShowNewJob(false);
                  }}
                />
              )}
            </div>

            {jobRows.length === 0 ? (
              <EmptyState
                title="No data generation jobs"
                description="Create a job to generate test datasets from your corpus."
              />
            ) : (
              <DataTable columns={jobColumns} data={jobRows} />
            )}
          </>
        ) : adapterRows.length === 0 ? (
          <EmptyState
            title="No LLM adapters configured"
            description="Add an LLM adapter to power test generation and evaluation."
          />
        ) : (
          <DataTable columns={adapterColumns} data={adapterRows} />
        )}
      </View>
    </Flex>
  );
}

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
        svg={<Icons.FileTextOutline />}
        color="var(--ac-global-text-color-500)"
      />
      <Heading level={3}>{title}</Heading>
      <Text color="text-500">{description}</Text>
    </Flex>
  );
}

function DataTable<T extends { id: string }>({
  columns,
  data,
}: {
  columns: ColumnDef<T>[];
  data: T[];
}) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <table css={tableCSS}>
      <thead>
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <th key={header.id} style={{ width: header.getSize() }}>
                {header.isPlaceholder
                  ? null
                  : flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id}>
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
