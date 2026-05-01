from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import anthropic

DATA_DIR = Path("data")
TASKS_FILE = Path("tasks/benchmark_tasks_v2.csv")
LOGS_DIR = Path("logs")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Conditions to judge
CONDITIONS = {
    "no_rag":    "logs/run_full_run_01_no_rag.jsonl",
    "rag_opt":   "logs/run_full_run_01_rag_opt.jsonl",
    "rag_forced":"logs/run_full_run_01_rag_forced.jsonl",
}

JUDGE_MODEL = "claude-sonnet-4-5"  # different from haiku used in experiments


JUDGE_SYSTEM = """You are an expert evaluator for a university student assistant system.
Your task is to score the quality of an agent's answer against a gold standard answer.

You will be given:
- A question asked by a student
- The gold standard answer (what the correct answer should say)
- The agent's answer to evaluate

Score the agent's answer on three dimensions:

CORRECTNESS (0, 1, or 2):
2 = Fully correct. All key facts from the gold answer are present. No factual errors.
1 = Partially correct. Some key facts present but missing important conditions, exceptions, or contains one minor factual error.
0 = Incorrect. Wrong facts, missing the core answer, or refuses to answer when an answer is possible.

HALLUCINATION (0 or 1):
0 = No hallucination. All specific claims (numbers, rules, deadlines, procedures) are supported by the gold answer or are reasonable inferences.
1 = Hallucination present. Contains a specific factual claim (number, rule, deadline, procedure) that contradicts the gold answer or cannot be verified.

GROUNDING (0, 1, 2, or NA):
Only score this for retrieval-required tasks (where retrieval_required = yes).
For retrieval-unnecessary tasks, output NA.
2 = Answer clearly cites or quotes from documents. Specific document language or facts are referenced.
1 = Answer appears document-grounded but without explicit citation or direct quotes.
0 = Answer appears to come from model memory, not documents.
NA = Task does not require retrieval.

Respond in JSON only, with this exact format:
{
  "correctness": <0, 1, or 2>,
  "hallucination": <0 or 1>,
  "grounding": <0, 1, 2, or "NA">,
  "correctness_reason": "<one sentence explaining the correctness score>",
  "hallucination_reason": "<one sentence, or 'None detected' if 0>"
}"""


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(path: str) -> dict[str, dict]:
    with open(path) as f:
        return {row["task_id"]: row for row in csv.DictReader(f)}


def judge_answer(
    client: anthropic.Anthropic,
    question: str,
    gold_answer: str,
    agent_answer: str,
    retrieval_required: str,
    task_id: str,
) -> dict:
    user_msg = f"""Task ID: {task_id}
Retrieval required: {retrieval_required}

QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

AGENT ANSWER:
{agent_answer}

Please score the agent's answer."""

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [warn] JSON parse failed for {task_id}, raw: {raw[:100]}")
        return {
            "correctness": -1,
            "hallucination": -1,
            "grounding": "ERROR",
            "correctness_reason": "parse error",
            "hallucination_reason": "parse error",
        }


def main():
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    tasks = load_tasks(TASKS_FILE)

    all_results = []

    for condition, log_path in CONDITIONS.items():
        print(f"\n=== Judging condition: {condition} ===")
        runs = load_jsonl(log_path)

        for run in runs:
            tid = run["task_id"]
            task = tasks.get(tid)
            if not task:
                print(f"  [skip] {tid} not in benchmark")
                continue

            print(f"  {tid}...", end=" ", flush=True)

            scores = judge_answer(
                client=client,
                question=task["question"],
                gold_answer=task["gold_answer"],
                agent_answer=run["answer"],
                retrieval_required=task["retrieval_required"],
                task_id=tid,
            )

            result = {
                "task_id": tid,
                "condition": condition,
                "group": task["group"],
                "retrieval_required": task["retrieval_required"],
                "retrieval_used": run["retrieval_used"],
                "tools_used": run["tools_used"],
                "num_retrieval_calls": run["num_retrieval_calls"],
                "judge_correctness": scores.get("correctness"),
                "judge_hallucination": scores.get("hallucination"),
                "judge_grounding": scores.get("grounding"),
                "judge_correctness_reason": scores.get("correctness_reason", ""),
                "judge_hallucination_reason": scores.get("hallucination_reason", ""),
                "answer_preview": run["answer"][:200],
            }
            all_results.append(result)
            print(f"correct={scores.get('correctness')} halluc={scores.get('hallucination')}")

            # Be polite to the API — small delay
            time.sleep(0.3)

    # Save full results
    out_jsonl = OUTPUT_DIR / "llm_judge_results.jsonl"
    with out_jsonl.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_results)} judgements to {out_jsonl}")

    # Print summary table
    print("\n=== SUMMARY ===")
    for condition in CONDITIONS:
        subset = [r for r in all_results if r["condition"] == condition]
        valid = [r for r in subset if isinstance(r["judge_correctness"], int) and r["judge_correctness"] >= 0]
        if not valid:
            continue
        mean_c = sum(r["judge_correctness"] for r in valid) / (len(valid) * 2)
        mean_h = sum(r["judge_hallucination"] for r in valid if isinstance(r["judge_hallucination"], int)) / len(valid)
        print(f"{condition}: correctness={mean_c:.2f} hallucination_rate={mean_h:.2f} (n={len(valid)})")

    # Save CSV for easy analysis
    out_csv = OUTPUT_DIR / "llm_judge_results.csv"
    if all_results:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"Saved CSV to {out_csv}")


if __name__ == "__main__":
    main()