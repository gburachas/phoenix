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

import type { CorpusSamplingConfig } from "./CorpusSamplingForm";
import { CorpusSamplingForm } from "./CorpusSamplingForm";

type LLMAdapterOption = {
  id: number;
  name: string;
  provider: string;
  modelName: string;
  canGenerate: boolean;
  canJudge: boolean;
  canEmbed: boolean;
};

const builderCSS = css`
  display: flex;
  flex-direction: column;
  gap: var(--ac-global-dimension-size-200);
  padding: var(--ac-global-dimension-size-200);
`;

const stepCSS = css`
  display: flex;
  flex-direction: column;
  gap: var(--ac-global-dimension-size-100);
  padding: var(--ac-global-dimension-size-200);
  border: 1px solid var(--ac-global-color-grey-300);
  border-radius: var(--ac-global-rounding-medium);
  background: var(--ac-global-background-color-light);
`;

const stepHeaderCSS = css`
  display: flex;
  align-items: center;
  gap: var(--ac-global-dimension-size-100);
`;

const stepNumberCSS = css`
  width: 24px;
  height: 24px;
  border-radius: 12px;
  background: var(--ac-global-color-primary);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
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
  width: 120px;
`;

const successCSS = css`
  padding: var(--ac-global-dimension-size-100) var(--ac-global-dimension-size-200);
  border: 1px solid var(--ac-global-color-green-700);
  border-radius: var(--ac-global-rounding-small);
  background: var(--ac-global-color-green-100, #e6f7e6);
  color: var(--ac-global-color-green-900, #1a7a1a);
  font-size: var(--ac-global-dimension-font-size-75);
`;

type DataGenPipelineBuilderProps = {
  onJobCreated?: () => void;
};

export function DataGenPipelineBuilder({
  onJobCreated,
}: DataGenPipelineBuilderProps) {
  const [adapters, setAdapters] = useState<LLMAdapterOption[]>([]);
  const [corpusConfig, setCorpusConfig] = useState<CorpusSamplingConfig | null>(
    null
  );
  const [jobName, setJobName] = useState("");
  const [testsetAdapterId, setTestsetAdapterId] = useState<string>("");
  const [transformAdapterId, setTransformAdapterId] = useState<string>("");
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxTokens, setMaxTokens] = useState<number>(1024);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Fetch adapters on mount
  useEffect(() => {
    let cancelled = false;
    const fetchAdapters = async () => {
      try {
        const resp = await fetch("/v1/llm-adapters");
        if (!resp.ok) return;
        const json = await resp.json();
        if (!cancelled) {
          setAdapters(
            (json.data || []).map((a: Record<string, unknown>) => ({
              id: a.id,
              name: a.name,
              provider: a.provider,
              modelName: a.model_name,
              canGenerate: a.can_generate,
              canJudge: a.can_judge,
              canEmbed: a.can_embed,
            }))
          );
        }
      } catch {
        // silent - adapters are optional
      }
    };
    fetchAdapters();
    return () => {
      cancelled = true;
    };
  }, []);

  const generationAdapters = adapters.filter((a) => a.canGenerate);

  const handleCreateJob = useCallback(async () => {
    if (!corpusConfig || !jobName.trim()) {
      setError("Please provide a job name and select a corpus source.");
      return;
    }
    setIsSubmitting(true);
    setError(null);
    setSuccess(null);
    try {
      const body: Record<string, unknown> = {
        name: jobName.trim(),
        corpus_source: corpusConfig.sourceName,
        corpus_config: {
          source_type: corpusConfig.sourceType,
          location: corpusConfig.location,
        },
        sampling_strategy: corpusConfig.strategy,
        sample_size: corpusConfig.sampleSize,
        llm_config: {
          temperature,
          max_tokens: maxTokens,
        },
      };
      if (testsetAdapterId) {
        body.testset_llm_adapter_id = Number(testsetAdapterId);
      }
      if (transformAdapterId) {
        body.transform_llm_adapter_id = Number(transformAdapterId);
      }

      const resp = await fetch("/v1/data-generation/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(errText || `Create failed: ${resp.status}`);
      }
      const json = await resp.json();
      setSuccess(
        `Job "${json.data.name}" created (ID: ${json.data.id}). Status: ${json.data.status}`
      );
      setJobName("");
      onJobCreated?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create job");
    } finally {
      setIsSubmitting(false);
    }
  }, [
    corpusConfig,
    jobName,
    testsetAdapterId,
    transformAdapterId,
    temperature,
    maxTokens,
    onJobCreated,
  ]);

  return (
    <div css={builderCSS}>
      <Heading level={3}>Create Data Generation Job</Heading>

      {/* Job name */}
      <div>
        <label css={labelCSS}>Job Name</label>
        <input
          css={inputCSS}
          style={{ width: "100%" }}
          type="text"
          placeholder="e.g., RAG testset v1"
          value={jobName}
          onChange={(e) => setJobName(e.target.value)}
        />
      </div>

      {/* Step 1: Corpus Sampling */}
      <div css={stepCSS}>
        <div css={stepHeaderCSS}>
          <span css={stepNumberCSS}>1</span>
          <Text>Corpus Sampling</Text>
        </div>
        <CorpusSamplingForm onConfigChange={setCorpusConfig} />
      </div>

      {/* Step 2: Test-set Generation LLM */}
      <div css={stepCSS}>
        <div css={stepHeaderCSS}>
          <span css={stepNumberCSS}>2</span>
          <Text>Test-set Generation LLM</Text>
        </div>
        <div>
          <label css={labelCSS}>LLM Adapter (for question/answer generation)</label>
          <select
            css={selectCSS}
            value={testsetAdapterId}
            onChange={(e) => setTestsetAdapterId(e.target.value)}
          >
            <option value="">— None (skip) —</option>
            {generationAdapters.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name} ({a.provider}/{a.modelName})
              </option>
            ))}
          </select>
        </div>
        <Flex direction="row" gap="size-200">
          <div>
            <label css={labelCSS}>Temperature</label>
            <input
              css={inputCSS}
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
            />
          </div>
          <div>
            <label css={labelCSS}>Max Tokens</label>
            <input
              css={inputCSS}
              type="number"
              min={1}
              max={16384}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
            />
          </div>
        </Flex>
      </div>

      {/* Step 3: Optional Transform LLM */}
      <div css={stepCSS}>
        <div css={stepHeaderCSS}>
          <span css={stepNumberCSS}>3</span>
          <Text>Optional Transform LLM</Text>
        </div>
        <div>
          <label css={labelCSS}>Transform Adapter (optional, for rephrasing/augmentation)</label>
          <select
            css={selectCSS}
            value={transformAdapterId}
            onChange={(e) => setTransformAdapterId(e.target.value)}
          >
            <option value="">— None (skip) —</option>
            {generationAdapters.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name} ({a.provider}/{a.modelName})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Submit */}
      {error && <Text color="danger">{error}</Text>}
      {success && <div css={successCSS}>{success}</div>}

      <Button
        variant="primary"
        size="M"
        onClick={handleCreateJob}
        isDisabled={isSubmitting || !corpusConfig || !jobName.trim()}
      >
        {isSubmitting ? "Creating..." : "Create Job"}
      </Button>
    </div>
  );
}
