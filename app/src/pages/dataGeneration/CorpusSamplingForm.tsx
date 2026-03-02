import { css } from "@emotion/react";
import { useCallback, useEffect, useState } from "react";

import {
  Button,
  Flex,
  Heading,
  Icon,
  Icons,
  Text,
  View,
} from "@phoenix/components";

type CorpusSource = {
  name: string;
  source_type: string;
  location: string;
  doc_count: number | null;
  metadata: Record<string, unknown>;
};

type SampledDoc = {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
};

const formCSS = css`
  display: flex;
  flex-direction: column;
  gap: var(--ac-global-dimension-size-200);
  padding: var(--ac-global-dimension-size-200);
  border: 1px solid var(--ac-global-color-grey-300);
  border-radius: var(--ac-global-rounding-medium);
  background: var(--ac-global-background-color-light);
`;

const labelCSS = css`
  font-size: var(--ac-global-dimension-font-size-75);
  font-weight: 600;
  color: var(--ac-global-text-color-700);
  margin-bottom: 4px;
`;

const selectCSS = css`
  padding: 6px 10px;
  border: 1px solid var(--ac-global-color-grey-400);
  border-radius: var(--ac-global-rounding-small);
  background: var(--ac-global-input-field-background-color);
  color: var(--ac-global-text-color-900);
  font-size: var(--ac-global-dimension-font-size-100);
`;

const inputCSS = css`
  padding: 6px 10px;
  border: 1px solid var(--ac-global-color-grey-400);
  border-radius: var(--ac-global-rounding-small);
  background: var(--ac-global-input-field-background-color);
  color: var(--ac-global-text-color-900);
  font-size: var(--ac-global-dimension-font-size-100);
  width: 100px;
`;

const radioGroupCSS = css`
  display: flex;
  gap: var(--ac-global-dimension-size-200);
  align-items: center;
`;

const radioLabelCSS = css`
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  font-size: var(--ac-global-dimension-font-size-75);
  color: var(--ac-global-text-color-700);
`;

const previewBoxCSS = css`
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--ac-global-color-grey-300);
  border-radius: var(--ac-global-rounding-small);
  padding: var(--ac-global-dimension-size-100);
  font-family: monospace;
  font-size: 12px;
  background: var(--ac-global-background-color-dark);
  color: var(--ac-global-text-color-700);
`;

export type CorpusSamplingConfig = {
  sourceName: string;
  sourceType: string;
  strategy: string;
  sampleSize: number;
  location?: string;
};

type CorpusSamplingFormProps = {
  onConfigChange?: (config: CorpusSamplingConfig | null) => void;
};

export function CorpusSamplingForm({ onConfigChange }: CorpusSamplingFormProps) {
  const [sources, setSources] = useState<CorpusSource[]>([]);
  const [selectedSource, setSelectedSource] = useState<string>("");
  const [strategy, setStrategy] = useState<string>("random");
  const [sampleSize, setSampleSize] = useState<number>(50);
  const [previewDocs, setPreviewDocs] = useState<SampledDoc[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch sources on mount
  useEffect(() => {
    let cancelled = false;
    const fetchSources = async () => {
      setIsLoading(true);
      try {
        const resp = await fetch("/v1/corpus/sources");
        if (!resp.ok) throw new Error(`Failed to fetch sources: ${resp.status}`);
        const json = await resp.json();
        if (!cancelled) {
          setSources(json.data || []);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load sources");
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };
    fetchSources();
    return () => {
      cancelled = true;
    };
  }, []);

  // Notify parent of config changes
  useEffect(() => {
    if (!selectedSource) {
      onConfigChange?.(null);
      return;
    }
    const source = sources.find((s) => s.name === selectedSource);
    if (!source) {
      onConfigChange?.(null);
      return;
    }
    onConfigChange?.({
      sourceName: source.name,
      sourceType: source.source_type,
      strategy,
      sampleSize,
      location: source.location,
    });
  }, [selectedSource, strategy, sampleSize, sources, onConfigChange]);

  const handlePreview = useCallback(async () => {
    const source = sources.find((s) => s.name === selectedSource);
    if (!source) return;
    setIsPreviewing(true);
    setError(null);
    try {
      const resp = await fetch("/v1/corpus/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_name: source.name,
          source_type: source.source_type,
          strategy,
          sample_size: Math.min(sampleSize, 5), // Preview limited to 5
          location: source.location,
        }),
      });
      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(errText || `Preview failed: ${resp.status}`);
      }
      const json = await resp.json();
      setPreviewDocs(json.data || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Preview failed");
    } finally {
      setIsPreviewing(false);
    }
  }, [selectedSource, sources, strategy, sampleSize]);

  return (
    <div css={formCSS}>
      <Heading level={4}>Corpus Sampling</Heading>

      {/* Source selector */}
      <div>
        <label css={labelCSS}>Corpus Source</label>
        {isLoading ? (
          <Text color="text-500">Loading sources...</Text>
        ) : (
          <select
            css={selectCSS}
            value={selectedSource}
            onChange={(e) => setSelectedSource(e.target.value)}
          >
            <option value="">— Select a source —</option>
            {sources.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name} ({s.source_type}
                {s.doc_count != null ? `, ${s.doc_count} docs` : ""})
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Strategy selector */}
      <div>
        <label css={labelCSS}>Sampling Strategy</label>
        <div css={radioGroupCSS}>
          {["random", "head", "tail"].map((s) => (
            <label key={s} css={radioLabelCSS}>
              <input
                type="radio"
                name="strategy"
                value={s}
                checked={strategy === s}
                onChange={() => setStrategy(s)}
              />
              {s === "random"
                ? "Random (entire corpus)"
                : s === "head"
                  ? "First N"
                  : "Last N"}
            </label>
          ))}
        </div>
        {strategy === "random" && (
          <Text
            color="text-500"
            elementType="p"
          >
            Samples uniformly from the entire corpus regardless of sample size.
          </Text>
        )}
      </div>

      {/* Sample size */}
      <div>
        <label css={labelCSS}>Sample Size</label>
        <Flex direction="row" alignItems="center" gap="size-100">
          <input
            css={inputCSS}
            type="number"
            min={1}
            max={10000}
            value={sampleSize}
            onChange={(e) => setSampleSize(Number(e.target.value) || 50)}
          />
          <input
            type="range"
            min={1}
            max={500}
            value={Math.min(sampleSize, 500)}
            onChange={(e) => setSampleSize(Number(e.target.value))}
            style={{ flex: 1 }}
          />
        </Flex>
      </div>

      {/* Preview */}
      <Flex direction="row" gap="size-100" alignItems="center">
        <Button
          variant="default"
          size="S"
          onClick={handlePreview}
          isDisabled={!selectedSource || isPreviewing}
        >
          {isPreviewing ? "Sampling..." : "Preview Sample"}
        </Button>
        {previewDocs.length > 0 && (
          <Text color="text-500">
            Showing {previewDocs.length} sample doc(s)
          </Text>
        )}
      </Flex>

      {error && (
        <Text color="danger">{error}</Text>
      )}

      {previewDocs.length > 0 && (
        <div css={previewBoxCSS}>
          {previewDocs.map((doc) => (
            <div key={doc.id} style={{ marginBottom: 8 }}>
              <strong>{doc.metadata?.filename ?? doc.id}</strong>
              <pre style={{ margin: "4px 0", whiteSpace: "pre-wrap" }}>
                {doc.content.slice(0, 300)}
                {doc.content.length > 300 ? "..." : ""}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
