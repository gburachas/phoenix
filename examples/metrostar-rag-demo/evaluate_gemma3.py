#!/usr/bin/env python3
"""
Evaluate Gemma 3:4B performance on RAG tasks with Phoenix tracing
"""
import os
import sys
import time
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure logging to suppress Phoenix 405 errors (non-blocking)
class Phoenix405Filter(logging.Filter):
    def filter(self, record):
        return "Failed to export span batch" not in record.getMessage()

# Apply filter to OpenTelemetry loggers
for logger_name in ['opentelemetry.sdk.trace.export', 'opentelemetry.exporter.otlp.proto.http.trace_exporter']:
    logger = logging.getLogger(logger_name)
    logger.addFilter(Phoenix405Filter())

# Phoenix instrumentation with NEW project name
from phoenix.otel import register
tracer_provider = register(
    project_name="evaluate_gemma3",  # New dedicated project
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:8085"),
)

print("="*80)
print("GEMMA 3:4B EVALUATION - PHOENIX TRACING ENABLED")
print("="*80)
print(f"Phoenix Project: evaluate_gemma3")
print(f"Phoenix URL: http://localhost:8085")
print("="*80 + "\n")

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# RAG setup (same as before)
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI
from typing import List

class OllamaEmbedding(BaseEmbedding):
    model_name: str
    api_base: str
    
    def __init__(self, model: str, api_base: str, **kwargs):
        super().__init__(model_name=model, api_base=api_base, **kwargs)
    
    @property
    def _client(self):
        if not hasattr(self, '_openai_client'):
            self._openai_client = OpenAI(base_url=self.api_base, api_key="ollama")
        return self._openai_client
    
    def _get_query_embedding(self, query: str) -> List[float]:
        response = self._client.embeddings.create(model=self.model_name, input=query)
        return response.data[0].embedding
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

# Setup services
print("Setting up RAG system...")
client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(client=client, collection_name="predictive_coding_paper")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

llm = Ollama(model="gemma3:4b", base_url="http://localhost:8090", temperature=0.7)
embed_model = OllamaEmbedding(model="mxbai-embed-large:335m", api_base="http://localhost:8089/v1")
Settings.llm = llm
Settings.embed_model = embed_model

index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
query_engine = index.as_query_engine(similarity_top_k=3)

print("✓ RAG system ready\n")

# Test queries to evaluate Gemma 3:4B
test_queries = [
    {
        "category": "Main Concept",
        "query": "What is Memory-augmented Dense Predictive Coding?",
        "expected": "Should explain the core architecture and memory mechanism"
    },
    {
        "category": "Technical Details",
        "query": "How does the predictive attention mechanism work?",
        "expected": "Should describe the attention over compressed memories"
    },
    {
        "category": "Methodology",
        "query": "What training approach does the paper use?",
        "expected": "Should mention contrastive learning and self-supervised learning"
    },
    {
        "category": "Results",
        "query": "What are the main experimental results?",
        "expected": "Should discuss performance on video representation tasks"
    },
    {
        "category": "Comparison",
        "query": "How does MemDPC compare to Dense Predictive Coding?",
        "expected": "Should explain the improvement from adding memory"
    }
]

print("="*80)
print("GEMMA 3:4B EVALUATION ON RAG TASKS")
print("="*80)
print(f"\nTesting {len(test_queries)} queries...")
print("All responses will be traced in Phoenix: http://localhost:8085\n")

results = []

for i, test in enumerate(test_queries, 1):
    print(f"\n[{i}/{len(test_queries)}] {test['category']}")
    print("-" * 80)
    print(f"Query: {test['query']}")
    print(f"Expected: {test['expected']}")
    print("\nProcessing...")
    
    start_time = time.time()
    response = query_engine.query(test['query'])
    end_time = time.time()
    
    response_time = end_time - start_time
    
    print(f"\n✓ Response ({response_time:.2f}s):")
    print(response.response)
    
    # Get source nodes for analysis
    source_count = len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
    avg_score = sum(node.score for node in response.source_nodes) / len(response.source_nodes) if source_count > 0 else 0
    
    results.append({
        "category": test['category'],
        "query": test['query'],
        "response": response.response,
        "response_time": response_time,
        "source_count": source_count,
        "avg_relevance": avg_score
    })
    
    time.sleep(1)  # Brief pause between queries

# Summary
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)

total_time = sum(r['response_time'] for r in results)
avg_time = total_time / len(results)

print(f"\nTotal queries: {len(results)}")
print(f"Total time: {total_time:.2f}s")
print(f"Average response time: {avg_time:.2f}s")
print(f"\nResponse times by category:")
for r in results:
    print(f"  {r['category']:20s}: {r['response_time']:5.2f}s (relevance: {r['avg_relevance']:.3f})")

print("\n" + "="*80)
print("✓ EVALUATION COMPLETE - CHECK PHOENIX UI")
print("="*80)
print(f"\nURL: http://localhost:8085")
print(f"Project: evaluate_gemma3")
print(f"\nAll {len(results)} queries have been traced!")
print("\nIn Phoenix UI, you can:")
print("  - View all 5 RAG queries in the Traces tab")
print("  - Compare response times across queries")
print("  - Analyze retrieval quality (relevance scores)")
print("  - See LLM token usage for Gemma 3:4B")
print("  - Debug slow operations (first query vs subsequent)")
print("  - Export traces for further analysis")
print("\nNext: Run evaluate_with_gpt5_judge.py to add LLM evaluations!")
print("="*80)
