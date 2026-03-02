#!/usr/bin/env python3
"""Phoenix RAG Evaluation with Gemma 3:4B via Ollama.

This script demonstrates:
1. Building a RAG application using Gemma 3:4B (via Ollama) as the LLM
2. Tracing the application with Phoenix
3. Evaluating responses using Azure OpenAI (GPT-5 Chat) as the Judge LLM
4. Running experiments and analyzing results

Based on the LLM Ops tutorial but adapted for local Gemma 3:4B model.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

# Load environment variables FIRST
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print("✓ Loaded configuration from .env\n")
else:
    print("❌ .env file not found")
    exit(1)

# Setup Phoenix Observability BEFORE any model imports
print("=" * 70)
print("Phoenix RAG Evaluation Setup - Gemma 3:4B with Azure Judge")
print("=" * 70)

try:
    from phoenix.otel import register
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    
    # Register Phoenix tracer
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:8085")
    project_name = "gemma3-rag-eval"
    
    print(f"\nRegistering Phoenix tracer...")
    print(f"   Phoenix endpoint: {phoenix_endpoint}")
    print(f"   Project name: {project_name}")
    
    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{phoenix_endpoint}/v1/traces",
        auto_instrument=False  # We'll instrument manually
    )
    
    # Instrument LlamaIndex and OpenAI
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    print("   ✓ Phoenix tracing enabled\n")
    
except ImportError as e:
    print(f"\n❌ Error: Phoenix instrumentation not available: {e}")
    print("Install with: conda activate arize && pip install arize-phoenix-otel openinference-instrumentation-llama-index openinference-instrumentation-openai")
    exit(1)

# Import required libraries
try:
    from llama_index.core import (
        Settings,
        VectorStoreIndex,
        SimpleDirectoryReader,
        Document,
    )
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
    from openai import OpenAI, AzureOpenAI
    from openinference.semconv.trace import SpanAttributes
    from opentelemetry import trace
    from phoenix.client import Client
    from openinference.instrumentation import suppress_tracing
    
except ImportError as e:
    print(f"❌ Error: Required packages not found: {e}")
    print("Install with: conda activate arize && pip install llama-index llama-index-llms-openai llama-index-embeddings-openai openai")
    exit(1)


def create_ollama_llm():
    """Create LlamaIndex-compatible LLM wrapper for Ollama (Gemma 3:4B)."""
    endpoint = os.getenv("OLLAMA_GEMMA3_ENDPOINT", "http://localhost:8090/v1")
    api_key = os.getenv("OLLAMA_GEMMA3_KEY", "ollama")
    
    print(f"\nOllama LLM Configuration:")
    print(f"   Endpoint: {endpoint}")
    print(f"   Model: gemma3:4b")
    
    return LlamaIndexOpenAI(
        model="gemma3:4b",
        api_base=endpoint,
        api_key=api_key,
    )


def create_sample_documents():
    """Create sample documents for RAG knowledge base."""
    documents = [
        Document(
            text="""
            Arize Phoenix is an open-source observability platform designed for machine learning 
            and LLM applications. It provides tracing, evaluation, and analysis capabilities for 
            AI systems. Phoenix helps teams understand model performance, debug issues, and 
            improve their ML applications through comprehensive monitoring and analytics.
            """,
            metadata={"source": "arize_overview.txt", "topic": "platform"}
        ),
        Document(
            text="""
            To use Phoenix tracing, you need to register your application using the 
            phoenix.otel.register() function. This should be done at the start of your application 
            before any LLM calls. You can specify a project name and endpoint. Phoenix supports 
            both local deployment (http://localhost:8085) and cloud deployment with Phoenix Cloud.
            """,
            metadata={"source": "tracing_setup.txt", "topic": "tracing"}
        ),
        Document(
            text="""
            LLM Evaluations (Evals) in Phoenix allow you to assess the quality of your LLM 
            applications using automated metrics. You can evaluate hallucinations, answer 
            correctness, relevance, and other criteria. Evals can be performed at both trace 
            level (evaluating entire request/response) and span level (evaluating individual 
            steps like retrieval or generation).
            """,
            metadata={"source": "evaluations.txt", "topic": "evaluations"}
        ),
        Document(
            text="""
            Phoenix supports experiments where you can test changes to your RAG application 
            systematically. Create a dataset of test queries, define your task function, and 
            specify evaluators. Phoenix will run your application on the dataset and measure 
            performance metrics, making it easy to compare different versions or configurations.
            """,
            metadata={"source": "experiments.txt", "topic": "experiments"}
        ),
        Document(
            text="""
            Gemma is a family of open-source language models developed by Google. Gemma 3:4B 
            is a 4-billion parameter model that can be run locally using tools like Ollama. 
            It's designed to be efficient and accessible while maintaining good performance on 
            many language tasks. Gemma models are OpenAI API compatible when served through Ollama.
            """,
            metadata={"source": "gemma_info.txt", "topic": "models"}
        ),
    ]
    
    print(f"   ✓ Created {len(documents)} sample documents\n")
    return documents


def setup_rag_system():
    """Setup the RAG system with Gemma 3:4B and Azure OpenAI embeddings."""
    print("\n" + "=" * 70)
    print("Setting up RAG System")
    print("=" * 70)
    
    # Configure LlamaIndex Settings
    Settings.llm = create_ollama_llm()
    
    # Use Azure OpenAI embeddings
    azure_endpoint = os.getenv("GPT5_MINI_ENDPOINT")
    api_key = os.getenv("GPT5_MINI_KEY")
    api_version = os.getenv("GPT5_MINI_API_VERSION")
    
    if not all([azure_endpoint, api_key, api_version]):
        print("❌ Error: Missing Azure OpenAI configuration for embeddings")
        exit(1)
    
    if azure_endpoint.endswith('/'):
        azure_endpoint = azure_endpoint.rstrip('/')
    
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment="text-embedding-3-small",
    )
    
    print(f"   Embedding Model: text-embedding-3-small (via Azure OpenAI)")
    
    # Create documents and index
    documents = create_sample_documents()
    
    print("   Building vector index...")
    index = VectorStoreIndex.from_documents(documents)
    print("   ✓ Vector index built\n")
    
    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=2)
    
    return query_engine


def create_azure_judge_llm():
    """Create Azure OpenAI Judge LLM for evaluations."""
    endpoint = os.getenv("GPT5_CHAT_ENDPOINT")
    api_key = os.getenv("GPT5_CHAT_KEY")
    deployment = os.getenv("GPT5_CHAT_DEPLOYMENT")
    api_version = os.getenv("GPT5_CHAT_API_VERSION")
    
    if not all([endpoint, api_key, deployment, api_version]):
        print("❌ Error: Missing Azure OpenAI configuration for Judge LLM")
        exit(1)
    
    if endpoint.endswith('/'):
        endpoint = endpoint.rstrip('/')
    
    print(f"\nAzure Judge LLM Configuration:")
    print(f"   Endpoint: {endpoint}")
    print(f"   Deployment: {deployment}")
    print(f"   API Version: {api_version}\n")
    
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    ), deployment


async def run_rag_queries(query_engine, queries):
    """Run queries through the RAG system with tracing."""
    print("\n" + "=" * 70)
    print("Running RAG Queries with Gemma 3:4B")
    print("=" * 70 + "\n")
    
    tracer = trace.get_tracer(__name__)
    results = []
    
    for query in tqdm(queries, desc="Processing queries"):
        with tracer.start_as_current_span("RAG Query") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "chain")
            span.set_attribute(SpanAttributes.INPUT_VALUE, query)
            
            try:
                response = query_engine.query(query)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
                
                print(f"\n{'─' * 70}")
                print(f"Query: {query}")
                print(f"Response: {response}")
                print(f"{'─' * 70}")
                
                results.append({
                    "query": query,
                    "response": str(response),
                    "success": True
                })
                
            except Exception as e:
                print(f"\n❌ Error processing query '{query}': {e}")
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Error: {e}")
                results.append({
                    "query": query,
                    "response": f"Error: {e}",
                    "success": False
                })
    
    print(f"\n✓ All queries processed and traced to Phoenix\n")
    return results


async def evaluate_with_azure_judge(judge_client, deployment, rag_results):
    """Manually evaluate RAG responses using Azure Judge LLM."""
    print("\n" + "=" * 70)
    print("Evaluating Responses with Azure Judge LLM")
    print("=" * 70 + "\n")
    
    evaluations = []
    
    for i, result in enumerate(rag_results, 1):
        if not result["success"]:
            continue
            
        print(f"\n--- Evaluation {i}/{len(rag_results)} ---")
        print(f"Query: {result['query']}")
        
        eval_prompt = f"""You are an expert evaluator. Evaluate this RAG response:

Question: {result['query']}
Answer: {result['response']}

Provide a brief evaluation covering:
1. Accuracy: Is the answer factually correct?
2. Relevance: Does it answer the question?
3. Quality: Is it clear and well-structured?
4. Score: Rate from 1-10

Keep your evaluation concise (2-3 sentences plus score).
"""
        
        try:
            with suppress_tracing():  # Don't trace the evaluation itself
                eval_response = judge_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator."},
                        {"role": "user", "content": eval_prompt}
                    ],
                    max_completion_tokens=500,
                    temperature=0.3,
                    model=deployment
                )
            
            evaluation = eval_response.choices[0].message.content
            print(f"\nJudge Evaluation:")
            print(f"{evaluation}\n")
            
            evaluations.append({
                "query": result['query'],
                "response": result['response'],
                "evaluation": evaluation
            })
            
        except Exception as e:
            print(f"❌ Error evaluating: {e}")
            evaluations.append({
                "query": result['query'],
                "response": result['response'],
                "evaluation": f"Error: {e}"
            })
    
    print(f"\n✓ Completed {len(evaluations)} evaluations\n")
    return evaluations


async def evaluate_traces(px_client, judge_llm, project_name):
    """Evaluate RAG responses using Azure Judge LLM."""
    print("\n" + "=" * 70)
    print("Evaluating Traces with Azure Judge LLM")
    print("=" * 70 + "\n")
    
    # Get spans from Phoenix
    print("   Fetching spans from Phoenix...")
    primary_df = await px_client.spans.get_spans_dataframe(project_identifier=project_name)
    
    if primary_df.empty:
        print("   ⚠️  No spans found. Make sure queries were run first.")
        return
    
    print(f"   ✓ Retrieved {len(primary_df)} spans\n")
    
    # Prepare trace-level data for evaluation
    spans_df = primary_df[
        [
            "name",
            "span_kind",
            "context.trace_id",
            "attributes.llm.input_messages",
            "attributes.llm.output_messages",
            "attributes.retrieval.documents",
        ]
    ]
    
    trace_df = (
        spans_df.groupby("context.trace_id")
        .agg({
            "attributes.llm.input_messages": lambda x: " ".join(x.dropna().astype(str)),
            "attributes.llm.output_messages": lambda x: " ".join(x.dropna().astype(str)),
            "attributes.retrieval.documents": lambda x: " ".join(x.dropna().astype(str)),
        })
        .rename(columns={
            "attributes.llm.input_messages": "input",
            "attributes.llm.output_messages": "output",
            "attributes.retrieval.documents": "reference",
        })
        .reset_index()
    )
    
    if trace_df.empty:
        print("   ⚠️  No trace data available for evaluation")
        return
    
    print(f"   Prepared {len(trace_df)} traces for evaluation\n")
    
    # Define evaluation templates
    HALLUCINATION_PROMPT = """
In this task, you will be presented with a query, a reference text and an answer. The answer is
generated to the question based on the reference text. The answer may contain false information.

[Query]: {{input}}
[Reference text]: {{reference}}
[Answer]: {{output}}

Is the answer factual or hallucinated based on the query and reference text?
Provide an explanation, then respond with LABEL: either "factual" or "hallucinated".
"""
    
    QA_PROMPT = """
You are given a question, an answer and reference text. Determine whether the answer correctly 
answers the question based on the reference text.

[Question]: {{input}}
[Reference]: {{reference}}
[Answer]: {{output}}

Provide an explanation, then respond with LABEL: either "correct" or "incorrect".
"""
    
    # Create evaluators using Azure Judge LLM
    print("   Creating evaluators...")
    hallucination_evaluator = create_classifier(
        name="hallucination",
        llm=judge_llm,
        prompt_template=HALLUCINATION_PROMPT,
        choices={"factual": 1.0, "hallucinated": 0.0},
    )
    
    qa_evaluator = create_classifier(
        name="q&a",
        llm=judge_llm,
        prompt_template=QA_PROMPT,
        choices={"correct": 1.0, "incorrect": 0.0},
    )
    
    # Run evaluations
    print("   Running evaluations with Azure Judge LLM...")
    with suppress_tracing():
        results_df = await async_evaluate_dataframe(
            dataframe=trace_df,
            evaluators=[hallucination_evaluator, qa_evaluator],
        )
    
    print(f"   ✓ Evaluations complete\n")
    
    # Display results
    print("\n" + "=" * 70)
    print("Evaluation Results Summary")
    print("=" * 70 + "\n")
    print(results_df[["hallucination.label", "hallucination.score", "q&a.label", "q&a.score"]])
    
    # Log annotations back to Phoenix
    print("\n   Logging evaluation results to Phoenix...")
    root_spans = primary_df[primary_df["parent_id"].isna()][["context.trace_id", "context.span_id"]]
    
    results_with_spans = pd.merge(
        results_df.reset_index(), root_spans, on="context.trace_id", how="left"
    ).set_index("context.span_id", drop=False)
    
    annotation_df = to_annotation_dataframe(dataframe=results_with_spans)
    
    hallucination_eval = annotation_df[annotation_df["annotation_name"] == "hallucination"].copy()
    qa_eval = annotation_df[annotation_df["annotation_name"] == "q&a"].copy()
    
    await px_client.annotations.log_span_annotations_dataframe(
        dataframe=hallucination_eval,
        annotator_kind="LLM",
    )
    
    await px_client.annotations.log_span_annotations_dataframe(
        dataframe=qa_eval,
        annotator_kind="LLM",
    )
    
    print("   ✓ Evaluation results logged to Phoenix\n")
    
    # Summary statistics
    hallucination_rate = (results_df["hallucination.label"] == "hallucinated").mean() * 100
    correctness_rate = (results_df["q&a.label"] == "correct").mean() * 100
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Hallucination Rate: {hallucination_rate:.1f}%")
    print(f"   Answer Correctness: {correctness_rate:.1f}%")
    print(f"   Total Traces Evaluated: {len(results_df)}\n")


async def evaluate_retrieval(px_client, judge_llm, project_name):
    """Evaluate retrieval quality (span-level evaluation)."""
    print("\n" + "=" * 70)
    print("Evaluating Retrieval Quality (Span-Level)")
    print("=" * 70 + "\n")
    
    # Get retrieval spans
    print("   Fetching retrieval spans from Phoenix...")
    primary_df = await px_client.spans.get_spans_dataframe(project_identifier=project_name)
    
    filtered_df = primary_df[
        (primary_df["span_kind"] == "RETRIEVER")
        & (primary_df["attributes.retrieval.documents"].notnull())
    ]
    
    if filtered_df.empty:
        print("   ⚠️  No retrieval spans found")
        return
    
    filtered_df = filtered_df.rename(
        columns={
            "attributes.input.value": "input",
            "attributes.retrieval.documents": "documents"
        }
    )
    
    print(f"   ✓ Retrieved {len(filtered_df)} retrieval spans\n")
    
    # Define retrieval relevancy template
    RAG_RELEVANCY_PROMPT = """
You are comparing a reference text to a question and determining if the reference text
contains information relevant to answering the question.

[Question]: {{input}}
[Reference text]: {{documents}}

Provide an explanation, then respond with LABEL: either "relevant" or "unrelated".
"""
    
    print("   Creating relevancy evaluator...")
    relevancy_evaluator = create_classifier(
        name="RAG Relevancy",
        llm=judge_llm,
        prompt_template=RAG_RELEVANCY_PROMPT,
        choices={"relevant": 1.0, "unrelated": 0.0},
    )
    
    # Run evaluation
    print("   Running relevancy evaluation...")
    with suppress_tracing():
        results_df = await async_evaluate_dataframe(
            dataframe=filtered_df,
            evaluators=[relevancy_evaluator],
        )
    
    print(f"   ✓ Retrieval evaluation complete\n")
    
    # Display results
    print("\n" + "=" * 70)
    print("Retrieval Relevancy Results")
    print("=" * 70 + "\n")
    print(results_df[["RAG Relevancy.label", "RAG Relevancy.score"]])
    
    # Log to Phoenix
    print("\n   Logging retrieval evaluation to Phoenix...")
    relevancy_eval_df = to_annotation_dataframe(dataframe=results_df)
    
    await px_client.annotations.log_span_annotations_dataframe(
        dataframe=relevancy_eval_df,
        annotator_kind="LLM",
    )
    
    print("   ✓ Retrieval evaluation logged to Phoenix\n")
    
    # Summary
    relevancy_rate = (results_df["RAG Relevancy.label"] == "relevant").mean() * 100
    print(f"\n📊 Retrieval Metrics:")
    print(f"   Relevancy Rate: {relevancy_rate:.1f}%")
    print(f"   Total Retrievals Evaluated: {len(results_df)}\n")


async def main():
    """Main execution flow."""
    print("\n" + "=" * 70)
    print("Phoenix RAG Evaluation - Gemma 3:4B with Azure Judge")
    print("=" * 70 + "\n")
    
    try:
        # Setup RAG system with Gemma 3:4B
        query_engine = setup_rag_system()
        
        # Define test queries
        queries = [
            "What is Arize Phoenix?",
            "How do I setup Phoenix tracing?",
            "What are LLM evaluations?",
            "How do experiments work in Phoenix?",
            "What is Gemma?",
            "How can I monitor my ML models?",
        ]
        
        # Run queries with tracing
        rag_results = await run_rag_queries(query_engine, queries)
        
        # Create Azure Judge LLM
        judge_client, deployment = create_azure_judge_llm()
        
        # Evaluate with Azure Judge (simple manual evaluation)
        evaluations = await evaluate_with_azure_judge(judge_client, deployment, rag_results)
        
        # Final summary
        print("\n" + "=" * 70)
        print("✓ EVALUATION COMPLETE")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   Total Queries: {len(rag_results)}")
        print(f"   Successful: {sum(1 for r in rag_results if r['success'])}")
        print(f"   Evaluated: {len(evaluations)}")
        print(f"\n📊 View traces in Phoenix UI: {os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:8085')}")
        print(f"   Project: gemma3-rag-eval")
        print("\n   All RAG traces are available in Phoenix!")
        print("   - View traces to see Gemma 3:4B RAG execution")
        print("   - Check evaluations from Azure Judge LLM above")
        print("   - Compare performance across different queries\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
