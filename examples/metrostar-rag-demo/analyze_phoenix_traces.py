#!/usr/bin/env python3
"""
Analyze Phoenix traces to get insights into RAG performance
"""
import requests
import json
from datetime import datetime

PHOENIX_URL = "http://localhost:8085"

def check_phoenix_status():
    """Check if Phoenix is accessible"""
    try:
        response = requests.get(PHOENIX_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("="*80)
    print("PHOENIX TRACE ANALYSIS")
    print("="*80)
    
    if not check_phoenix_status():
        print(f"\n❌ Phoenix UI not accessible at {PHOENIX_URL}")
        print("Start Phoenix with: docker-compose up -d phoenix")
        return
    
    print(f"\n✓ Phoenix UI accessible at {PHOENIX_URL}")
    
    print("\n" + "-"*80)
    print("HOW TO ANALYZE TRACES IN PHOENIX UI")
    print("-"*80)
    
    print("""
1. OPEN PHOENIX UI:
   http://localhost:8085
   
2. VIEW TRACES:
   - Click on "Traces" tab in the left sidebar
   - You should see traces from evaluate_gemma3.py (5 queries)
   
3. ANALYZE A TRACE:
   Click on any trace to see:
   - Total duration breakdown
   - Span hierarchy (embedding → retrieval → LLM)
   - Input/output at each step
   - Retrieved context chunks
   - LLM response generation time
   
4. COMPARE QUERIES:
   - Sort by duration to find slow queries
   - Check relevance scores in retrieval spans
   - Compare first query (7.6s) vs subsequent (~1s)
   
5. IDENTIFY BOTTLENECKS:
   Look at span durations:
   - Embedding generation: ~200ms expected
   - Qdrant retrieval: ~100ms expected
   - LLM generation: ~1-4s expected (varies by response length)
   
6. REVIEW RETRIEVED CONTEXT:
   - Check which document chunks were retrieved
   - Verify relevance scores (0.5-0.8 is good)
   - Ensure diverse content from the paper
   
7. MONITOR LLM PERFORMANCE:
   - Input token count
   - Output token count
   - Tokens per second
   - Response quality vs. speed trade-off
   
8. EXPORT DATA:
   - Download traces as JSON for custom analysis
   - Set up Phoenix evaluators for automated quality checks
   - Create dashboards for ongoing monitoring
""")
    
    print("-"*80)
    print("EXPECTED PERFORMANCE (from evaluate_gemma3.py)")
    print("-"*80)
    print("""
Query Type              | Response Time | Relevance | Notes
------------------------|---------------|-----------|------------------
Main Concept            | 7.60s         | 0.830     | First query (cold start)
Technical Details       | 1.30s         | 0.620     | Model warmed up
Methodology             | 0.95s         | 0.535     | Fast response
Results                 | 0.96s         | 0.496     | Fast response
Comparison              | 1.02s         | 0.818     | High relevance

Average (after warmup): 1.06s per query
""")
    
    print("-"*80)
    print("USEFUL PHOENIX FEATURES")
    print("-"*80)
    print("""
✓ Span visualization: See the full RAG pipeline
✓ Latency analysis: Identify slow components
✓ Input/Output inspection: Debug incorrect responses
✓ Metadata tracking: Filter by project, tags, etc.
✓ Error tracking: Catch failed operations
✓ Token counting: Monitor LLM usage
✓ Retrieval quality: Check relevance scores
✓ Time-series view: Track performance over time
""")
    
    print("\n" + "="*80)
    print(f"Open Phoenix UI: {PHOENIX_URL}")
    print("="*80)

if __name__ == "__main__":
    main()
