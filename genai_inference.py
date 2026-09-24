"""
genai_inference.py - Track 2: Generative AI Pipeline for Drug Review Evaluation

Features:
- Async concurrent LLM inference using aiohttp and asyncio.
- Supports Google Gemini API (gemini-2.5-flash / gemini-1.5-flash) and OpenAI API.
- Alternative 1: Few-Shot Chain-of-Thought (CoT) Prompting with step-by-step reasoning.
- Alternative 2: Zero-Shot Persona-Driven Prompting with explicit clinical rating rubrics.
- Robust JSON parsing and error recovery fallback logic.
- Excel exports: Output_Alternative_1.xlsx & Output_Alternative_2.xlsx.
- Evaluation metrics: 10-Class Exact Match Accuracy, Off-by-1 Accuracy, and Quadratic Weighted Kappa (QWK).
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import aiohttp
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error

# Configure UTF-8 encoding for standard output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

import random


# ---------------------------------------------------------------------------
# Prompt Templates & Engineering Strategies
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ALT1_COT = """You are a Senior Clinical Pharmacologist and Medical Evaluator.
Your task is to analyze patient drug reviews and evaluate the medication's effectiveness, safety, and sentiment.

RATING SCALE ANCHOR RULES (Scale 1 to 10):
- Patient ratings in medical reviews are strongly bimodal:
  - Exact 10: Complete symptom relief, life-changing efficacy, zero/negligible side effects. (Do NOT default to 8 or 9 if patient is highly satisfied!).
  - Exact 8-9: High efficacy with manageable minor side effects.
  - Exact 5-7: Moderate efficacy, partial relief, or noticeable side effects.
  - Exact 2-3: Low efficacy, high dissatisfaction, or severe cramping/nausea.
  - Exact 1: Complete treatment failure, unbearable side effects, dangerous toxicity, or "worst medicine ever". (Do NOT default to 2 or 3 if patient experienced total failure!).

You MUST follow a Chain-of-Thought (CoT) approach: explain your clinical reasoning BEFORE providing the final numerical rating, summary, and sentiment.

Examples:

Example 1 (Rating 10 Anchor):
Input: Drug: Mirtazapine | Condition: Depression | Review: "This medicine has saved my life. Thankfully I have had no side effects especially the most common - weight gain."
JSON Output:
{
  "reasoning": "Patient reports life-saving therapeutic efficacy for depression with no side effects. Full rating 10 applies.",
  "predicted_rating": 10,
  "generated_summary": "Life saving medication with zero severe side effects.",
  "identified_sentiment": "positive"
}

Example 2 (Rating 8 Anchor):
Input: Drug: Lisinopril | Condition: High Blood Pressure | Review: "Lowered my blood pressure from 150 to 120 within two weeks. Experienced slight dry cough in mornings, but tolerable."
JSON Output:
{
  "reasoning": "High efficacy in controlling blood pressure with minor manageable cough side effect.",
  "predicted_rating": 8,
  "generated_summary": "Effective blood pressure control with minor tolerable cough.",
  "identified_sentiment": "positive"
}

Example 3 (Rating 5 Anchor):
Input: Drug: Copper | Condition: Birth Control | Review: "Periods last 11 days and I'm in pain for about 15 days... I'd recommend this 100% to somebody who doesn't already have heavy periods but right now it just isn't for me."
JSON Output:
{
  "reasoning": "Patient experienced severe cramping requiring discontinuation, despite recommending to non-cramping candidates.",
  "predicted_rating": 5,
  "generated_summary": "Severe cramping and heavy periods despite non-hormonal benefit.",
  "identified_sentiment": "negative"
}

Example 4 (Rating 3 Anchor):
Input: Drug: Metformin | Condition: Diabetes | Review: "Controls my blood sugar somewhat, but the severe nausea and diarrhea every single day make it almost unmanageable."
JSON Output:
{
  "reasoning": "Partial blood sugar control overshadowed by chronic daily gastrointestinal adverse reactions.",
  "predicted_rating": 3,
  "generated_summary": "Partial efficacy ruined by daily severe gastrointestinal distress.",
  "identified_sentiment": "negative"
}

Example 5 (Rating 1 Anchor):
Input: Drug: Trazodone | Condition: Insomnia | Review: "Started at 50mg, and felt WIRED. Bumped to 150, with the same lack of effect... LEAST effective I have ever come across."
JSON Output:
{
  "reasoning": "Complete therapeutic failure for insomnia and induced paradoxical hyperactivity across all doses. Full rating 1 applies.",
  "predicted_rating": 1,
  "generated_summary": "Ineffective sleep aid caused paradoxical hyperactivity.",
  "identified_sentiment": "negative"
}

OUTPUT FORMAT INSTRUCTION:
You MUST output ONLY a valid JSON object with the following exact keys:
{
  "reasoning": "string (1-2 clinical sentences)",
  "predicted_rating": integer (1 to 10 scale),
  "generated_summary": "string (STRICTLY MAXIMUM 10 WORDS)",
  "identified_sentiment": "string (strictly 'positive' or 'negative')"
}
"""

import html

SYSTEM_PROMPT_ALT2_PERSONA = """You are an Expert Clinical Pharmacologist and Drug Safety Specialist.
Your sole mission is to rigorously evaluate patient drug reviews based on strict clinical rating boundary criteria.

BIMODAL RATING BOUNDARY DEFINITIONS (Scale 1 to 10):
- Rating 10 (EXACT 10): Absolute therapeutic efficacy, complete symptom elimination, total patient satisfaction, zero side effects. Assign 10 directly when patient calls it "life-saving", "great", "excellent", or "outstanding"!
- Rating 8-9: High clinical efficacy, substantial symptom improvement, manageable or minor side effects. Assign 9 for short positive reviews like "Good", "Works well", or "Very good".
- Rating 6-7: Moderate efficacy, partial symptom relief, noticeable mild-to-moderate side effects.
- Rating 4-5: Low efficacy, minimal symptom improvement, annoying/moderate adverse effects, mixed outcome.
- Rating 2-3: Very poor efficacy, minimal relief, significant adverse effects, high patient dissatisfaction.
- Rating 1 (EXACT 1): Complete therapeutic failure, severe adverse reactions, dangerous toxicity, or unbearable side effects. Assign 1 directly when patient rates it worst ever or total failure!

SHORT REVIEW RULE:
For concise 1-4 word reviews (e.g. "Good", "Great!", "Works well", "Excellent"):
- If positive -> Assign Rating 9 or 10.
- If negative -> Assign Rating 1 or 2.

OUTPUT FORMAT INSTRUCTION:
Do not include chain-of-thought narrative outside the JSON structure.
You MUST output ONLY a valid JSON object with the following exact keys:
{
  "predicted_rating": integer (1 to 10 scale),
  "generated_summary": "string (STRICTLY MAXIMUM 10 WORDS)",
  "identified_sentiment": "string (strictly 'positive' or 'negative')"
}
"""


def preprocess_review_text(raw_review: str) -> str:
    """Clean HTML entities, newlines, and escape characters from review text."""
    if not isinstance(raw_review, str):
        return ""
    clean_text = html.unescape(raw_review)
    clean_text = clean_text.replace("\r\n", " ").replace("\n", " ").strip()
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text


def build_user_prompt(drug_name: str, condition: str, review: str, useful_count: int = 0) -> str:
    clean_rev = preprocess_review_text(review)
    return f"Drug: {drug_name}\nCondition: {condition}\nPatient Upvotes: {useful_count}\nReview: \"{clean_rev}\""


# ---------------------------------------------------------------------------
# Robust JSON Extraction & Parsing Fallbacks
# ---------------------------------------------------------------------------

def clean_and_truncate_summary(summary_raw: str, max_words: int = 10) -> str:
    """Clean summary text and enforce maximum word count limit."""
    if not summary_raw or not isinstance(summary_raw, str):
        return "No summary provided."
    summary_clean = re.sub(r"\s+", " ", summary_raw).strip()
    words = summary_clean.split(" ")
    if len(words) > max_words:
        summary_clean = " ".join(words[:max_words])
    return summary_clean


def normalize_sentiment(sentiment_raw: Any, predicted_rating: int) -> str:
    """Normalize sentiment string strictly to 'positive' or 'negative'."""
    if isinstance(sentiment_raw, str):
        s_lower = sentiment_raw.strip().lower()
        if "pos" in s_lower:
            return "positive"
        if "neg" in s_lower:
            return "negative"
    # Fallback based on rating threshold if ambiguous or missing
    return "positive" if predicted_rating >= 6 else "negative"


def parse_llm_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Robust JSON parser with regex extraction and error fallback logic.
    Handles DeepSeek-R1 <think>...</think> reasoning blocks.
    """
    cleaned_text = raw_text.strip()
    # Strip DeepSeek-R1 <think>...</think> internal reasoning blocks
    cleaned_text = re.sub(r"<think>[\s\S]*?</think>", "", cleaned_text).strip()

    # Try finding markdown JSON block ```json ... ``` or standard {...}
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned_text, re.IGNORECASE)
    if json_match:
        target_str = json_match.group(1)
    else:
        bracket_match = re.search(r"(\{[\s\S]*\})", cleaned_text)
        target_str = bracket_match.group(1) if bracket_match else cleaned_text

    try:
        data = json.loads(target_str)
    except Exception:
        # Emergency regex fallbacks if JSON fails completely
        rating_match = re.search(r'"predicted_rating"\s*:\s*(\d+)', cleaned_text)
        summary_match = re.search(r'"generated_summary"\s*:\s*"([^"]+)"', cleaned_text)
        sentiment_match = re.search(r'"identified_sentiment"\s*:\s*"([^"]+)"', cleaned_text)

        rating_val = int(rating_match.group(1)) if rating_match else 5
        summary_val = summary_match.group(1) if summary_match else "Evaluation completed."
        sentiment_val = sentiment_match.group(1) if sentiment_match else ("positive" if rating_val >= 6 else "negative")

        data = {
            "predicted_rating": rating_val,
            "generated_summary": summary_val,
            "identified_sentiment": sentiment_val,
        }

    # Validate and constrain predicted_rating
    try:
        rating = int(data.get("predicted_rating", 5))
        rating = max(1, min(10, rating))
    except (ValueError, TypeError):
        rating = 5

    # Validate summary
    summary = clean_and_truncate_summary(str(data.get("generated_summary", "")), max_words=10)

    # Validate sentiment
    sentiment = normalize_sentiment(data.get("identified_sentiment"), rating)

    return {
        "predicted_rating": rating,
        "generated_summary": summary,
        "identified_sentiment": sentiment,
    }


# ---------------------------------------------------------------------------
# Async API Worker Engine with Resilient Exponential Backoff & Rate Pacing
# ---------------------------------------------------------------------------

async def call_llm_api_single(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    provider: str,
    base_url: str,
    model: str,
    delay: float = 0.05,
    max_retries: int = 6,
) -> Dict[str, Any]:
    """Call LLM API (Gemini or OpenAI) with exponential backoff on 429/rate-limits."""
    async with semaphore:
        if delay > 0:
            await asyncio.sleep(delay)
        for attempt in range(max_retries):
            try:
                if provider.lower() == "gemini":
                    # Google Gemini REST API (v1beta generateContent)
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                    payload = {
                        "systemInstruction": {
                            "parts": [{"text": system_prompt}]
                        },
                        "contents": [
                            {"role": "user", "parts": [{"text": user_prompt}]}
                        ],
                        "generationConfig": {
                            "temperature": 0.1,
                            "responseMimeType": "application/json"
                        }
                    }
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status == 200:
                            res_json = await resp.json()
                            candidates = res_json.get("candidates", [])
                            if candidates and "content" in candidates[0]:
                                parts = candidates[0]["content"].get("parts", [])
                                if parts and "text" in parts[0]:
                                    return parse_llm_json_response(parts[0]["text"])
                        elif resp.status in (429, 500, 503, 504):
                            wait_time = (2 ** attempt) * 3.0 + random.uniform(1.0, 3.0)
                            print(f"⚠️ Rate limited (HTTP {resp.status}). Retrying in {wait_time:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await resp.text()
                            print(f"⚠️ Gemini API HTTP {resp.status}: {error_text[:120]}")
                            await asyncio.sleep(2.0)
                else:
                    # OpenAI REST API
                    url = f"{base_url.rstrip('/')}/chat/completions"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    }
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 150,
                    }
                    if "api.openai.com" in url:
                        payload["response_format"] = {"type": "json_object"}
                    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=90)) as resp:
                        if resp.status == 200:
                            res_json = await resp.json()
                            raw_content = res_json["choices"][0]["message"]["content"]
                            return parse_llm_json_response(raw_content)
                        elif resp.status in (429, 500, 503, 504):
                            wait_time = (2 ** attempt) * 3.0 + random.uniform(1.0, 3.0)
                            print(f"⚠️ OpenAI Rate limited (HTTP {resp.status}). Retrying in {wait_time:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await resp.text()
                            print(f"⚠️ OpenAI API HTTP {resp.status}: {error_text[:120]}")
                            await asyncio.sleep(2.0)
            except Exception as e:
                wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
                await asyncio.sleep(wait_time)

    # Return safe fallback if all retries fail
    return {
        "predicted_rating": 5,
        "generated_summary": "API request failed fallback.",
        "identified_sentiment": "positive",
    }


async def process_batch_strategy(
    df: pd.DataFrame,
    system_prompt: str,
    api_key: str,
    provider: str,
    base_url: str,
    model: str,
    concurrency: int,
    strategy_name: str,
    delay: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Process all dataset rows with incremental disk caching to support seamless resume.
    """
    semaphore = asyncio.Semaphore(concurrency)
    cache_filename = f".cache_{strategy_name.lower().replace(' ', '_').replace(':', '')}.json"

    # Load cached results if present
    cached_data: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            print(f"📦 Found existing cache ({len(cached_data)} items loaded from {cache_filename}).")
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")

    total_rows = len(df)
    results_map: Dict[int, Dict[str, Any]] = {}

    # Pre-fill from cache
    missing_indices = []
    for idx in range(total_rows):
        key_str = str(idx)
        if key_str in cached_data:
            results_map[idx] = cached_data[key_str]
        else:
            missing_indices.append(idx)

    print(f"🚀 {strategy_name}: {len(results_map)} cached, {len(missing_indices)} remaining to fetch (Model={model}, Concurrency={concurrency}, Delay={delay}s)...", flush=True)
    start_time = time.time()

    if missing_indices:
        async with aiohttp.ClientSession() as session:
            completed_count = len(results_map)

            async def fetch_row(idx: int) -> Tuple[int, Dict[str, Any]]:
                nonlocal completed_count
                row = df.iloc[idx]
                drug_name = str(row.get("drugName", "Unknown"))
                condition = str(row.get("condition", "General"))
                review = str(row.get("review", ""))
                useful_count = int(row.get("usefulCount", 0)) if pd.notna(row.get("usefulCount")) else 0
                user_prompt = build_user_prompt(drug_name, condition, review, useful_count=useful_count)

                res = await call_llm_api_single(
                    session=session,
                    semaphore=semaphore,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_key=api_key,
                    provider=provider,
                    base_url=base_url,
                    model=model,
                    delay=delay,
                )

                completed_count += 1
                if completed_count % 50 == 0 or completed_count == total_rows:
                    pct = (completed_count / total_rows) * 100.0
                    print(f"  Progress: [{completed_count}/{total_rows}] ({pct:.1f}%) processed...")

                return idx, res

            tasks = [fetch_row(idx) for idx in missing_indices]
            fetched = await asyncio.gather(*tasks)

            # Update results map and cache file
            for idx, res in fetched:
                results_map[idx] = res
                cached_data[str(idx)] = res

            # Save to cache on completion or update
            try:
                with open(cache_filename, "w", encoding="utf-8") as f:
                    json.dump(cached_data, f, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ Failed to save cache: {e}")

    elapsed = time.time() - start_time
    print(f"✅ Completed {strategy_name} in {elapsed:.2f}s.")

    # Return ordered results
    return [results_map[i] for i in range(total_rows)]


# ---------------------------------------------------------------------------
# Metrics Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate_and_report(df_result: pd.DataFrame, title: str) -> Dict[str, float]:
    """Calculate and format evaluation metrics against true ratings."""
    y_true = df_result["rating"].astype(int).values
    y_pred = df_result["predicted_rating"].astype(int).values

    exact_match = accuracy_score(y_true, y_pred) * 100.0
    off_by_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100.0
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(1, 11)))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n=================================================================")
    print(f"       {title.upper()} EVALUATION REPORT")
    print(f"=================================================================")
    print(f" Total Evaluated Samples:     {len(df_result)}")
    print(f"-----------------------------------------------------------------")
    print(f" 10-Class Exact Match Acc:    {exact_match:.2f}%")
    print(f" Off-by-1 Accuracy (<= 1):    {off_by_1:.2f}%")
    print(f" Quadratic Weighted Kappa:    {qwk:.4f}")
    print(f" Mean Absolute Error (MAE):   {mae:.4f}")
    print(f" Root Mean Sq Error (RMSE):   {rmse:.4f}")
    print(f"=================================================================\n")

    return {
        "ExactMatch": exact_match,
        "OffBy1": off_by_1,
        "QWK": qwk,
        "MAE": mae,
        "RMSE": rmse,
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Track 2: LLM Pipeline for UCI Drug Review Dataset")
    parser.add_argument("--test_path", type=str, default="strict_test.csv", help="Path to test CSV file")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of rows to sample for testing (e.g. 100)")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent API calls")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay in seconds between API calls")
    parser.add_argument("--provider", type=str, choices=["gemini", "openai"], default="openai", help="LLM Provider")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", "lm-studio"), help="API Key")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1", help="API Base URL")
    parser.add_argument("--model", type=str, default="qwen2.5-coder-7b-instruct", help="LLM Model Name")
    parser.add_argument("--strategy", type=str, choices=["alt1", "alt2", "both"], default="both", help="Strategy to run")
    parser.add_argument("--clear_cache", action="store_true", help="Clear disk cache to force fresh inference with updated prompts")
    args = parser.parse_args()

    if args.clear_cache:
        import glob
        cache_files = glob.glob(".cache*.json")
        for cf in cache_files:
            try:
                os.remove(cf)
                print(f"🗑️ Cleared cache file: {cf}")
            except Exception as e:
                print(f"⚠️ Could not delete {cf}: {e}")

    if not args.api_key:
        print("❌ ERROR: API key is missing. Pass --api_key or set GEMINI_API_KEY.")
        sys.exit(1)

    if not os.path.exists(args.test_path):
        print(f"❌ ERROR: Test dataset file not found at: {args.test_path}")
        sys.exit(1)

    print(f"📖 Reading dataset from {args.test_path}...")
    df_raw = pd.read_csv(args.test_path)

    # Required baseline columns
    required_cols = ["drugName", "condition", "review", "rating"]
    for col in required_cols:
        if col not in df_raw.columns:
            print(f"❌ ERROR: Required column '{col}' missing from {args.test_path}.")
            sys.exit(1)

    # Sample dataset if requested
    if args.sample_size and args.sample_size > 0:
        print(f"🔬 Sampling top {args.sample_size} rows for rapid verification...")
        df = df_raw.iloc[: args.sample_size].copy()
    else:
        df = df_raw.copy()

    # Define execution routines
    strategies_to_run = []
    if args.strategy in ["alt1", "both"]:
        strategies_to_run.append(("Alternative 1: Few-Shot CoT", SYSTEM_PROMPT_ALT1_COT, "Output_Alternative_1.xlsx"))
    if args.strategy in ["alt2", "both"]:
        strategies_to_run.append(("Alternative 2: Zero-Shot Persona", SYSTEM_PROMPT_ALT2_PERSONA, "Output_Alternative_2.xlsx"))

    metrics_summary = {}

    for strat_name, sys_prompt, out_excel in strategies_to_run:
        # Run async event loop
        parsed_llm_results = asyncio.run(
            process_batch_strategy(
                df=df,
                system_prompt=sys_prompt,
                api_key=args.api_key,
                provider=args.provider,
                base_url=args.base_url,
                model=args.model,
                concurrency=args.concurrency,
                strategy_name=strat_name,
                delay=args.delay,
            )
        )

        # Build clean output dataframe with required columns
        df_out = pd.DataFrame()
        df_out["drugName"] = df["drugName"].values
        df_out["condition"] = df["condition"].values
        df_out["review"] = df["review"].values
        df_out["rating"] = df["rating"].astype(int).values

        # Append LLM derived fields
        df_out["predicted_rating"] = [res["predicted_rating"] for res in parsed_llm_results]
        df_out["generated_summary"] = [res["generated_summary"] for res in parsed_llm_results]
        df_out["identified_sentiment"] = [res["identified_sentiment"] for res in parsed_llm_results]

        # Enforce exact requested output column order
        exact_columns = ["drugName", "condition", "review", "rating", "predicted_rating", "generated_summary", "identified_sentiment"]
        df_out = df_out[exact_columns]

        # Export to Excel
        print(f"💾 Exporting results to {out_excel}...")
        try:
            df_out.to_excel(out_excel, index=False)
            print(f"✅ File successfully saved: {out_excel}")
        except Exception as e:
            out_csv = out_excel.replace(".xlsx", ".csv")
            print(f"⚠️ Excel export warning ({e}). Saving CSV to {out_csv}...")
            df_out.to_csv(out_csv, index=False)
            print(f"✅ File successfully saved: {out_csv}")

        # Compute & report metrics
        metrics = evaluate_and_report(df_out, strat_name)
        metrics_summary[strat_name] = metrics

    # Comparative Summary Table
    if len(metrics_summary) > 1:
        print("\n=================================================================")
        print("          TRACK 2 PROMPT STRATEGIES COMPARATIVE SUMMARY           ")
        print("=================================================================")
        print(f"{'Strategy':<30} | {'Exact Match':<11} | {'Off-by-1':<10} | {'QWK':<6}")
        print("-" * 65)
        for strat_name, m in metrics_summary.items():
            print(f"{strat_name:<30} | {m['ExactMatch']:>10.2f}% | {m['OffBy1']:>9.2f}% | {m['QWK']:>6.4f}")
        print("=================================================================\n")


if __name__ == "__main__":
    main()
