import asyncio
import json
import time
import pandas as pd
from typing import List, Dict

# Apply nest_asyncio if running in environment where loop is already running, though standalone shouldn't need it.
import nest_asyncio
nest_asyncio.apply()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
)

from app.services.dialogue_service import dialogue_service

# Load LangChain/Ragas configuration 
# We need to make sure API keys are loaded
from app.core.config import settings
import os
if settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
if settings.GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY


async def evaluate_rag():
    print("Loading datasets...")
    # Read and concatenate multiple datasets to get enough diversity
    import glob
    csv_files = glob.glob("data/meddata/*QA.csv")
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Take a sample of 10 entries
    sample_size = 10
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    questions = df_sample['Question'].tolist()
    ground_truths = df_sample['Answer'].tolist()
    
    answers = [""] * len(questions)
    contexts = [[]] * len(questions)
    latencies = [0.0] * len(questions)
    
    print(f"Starting evaluation of {len(questions)} questions...")
    
    sem = asyncio.Semaphore(1) # Strict concurrency of 1 to prevent free-tier rate limits
    
    async def process_question(idx, q):
        async with sem:
            start_time = time.time()
            try:
                res = await dialogue_service.answer_question(query=q, top_k=3)
                latency = time.time() - start_time
                latencies[idx] = latency
                answers[idx] = res['answer']
                contexts[idx] = [source['snippet'] for source in res['sources']]
                await asyncio.sleep(2.5) # strict delay to keep requests under ~24 per minute
            except Exception as e:
                print(f"Error querying RAG for question {idx}: {e}")
                answers[idx] = "Error occurred"
                contexts[idx] = [""]
                latencies[idx] = 0.0

    tasks = [process_question(i, q) for i, q in enumerate(questions)]
    await asyncio.gather(*tasks)
            
    print("Creating Ragas dataset...")
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data_dict)
    
    print("Running Ragas evaluation metrics...")
    
    # Initialize explicit LLM (Groq) and Embeddings (Gemini) to bypass rate limits
    from langchain_groq import ChatGroq
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas.run_config import RunConfig
    
    eval_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=settings.GROQ_API_KEY)
    eval_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=settings.GEMINI_API_KEY)
    
    # Add a RunConfig to limit concurrency just in case Groq also gets rate limited
    eval_config = RunConfig(max_workers=2, max_retries=15)
    
    try:
        # Evaluate using standard metrics
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity],
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=eval_config
        )
        
        # Convert to pandas
        results_df = results.to_pandas()
        
        # Add latency to the results
        results_df['latency_sec'] = latencies
        
        # Calculate averages
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        eval_dict = dict(results)
        
        final_output = {
            "averages": {
                "faithfulness": eval_dict.get("faithfulness", 0.0),
                "answer_relevancy": eval_dict.get("answer_relevancy", 0.0),
                "context_precision": eval_dict.get("context_precision", 0.0),
                "context_recall": eval_dict.get("context_recall", 0.0),
                "answer_similarity": eval_dict.get("answer_similarity", 0.0),
                "latency_sec": avg_latency
            },
            "detailed_results": results_df.to_dict(orient="records")
        }
        
        output_path = "app/static/evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
            
        print(f"Evaluation complete. Results saved to {output_path}")
        print(f"Averages: {final_output['averages']}")
        
    except Exception as e:
        print(f"Error during Ragas evaluation: {e}")

if __name__ == "__main__":
    asyncio.run(evaluate_rag())
