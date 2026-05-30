"""
Run the tool-calling agent across the benchmark under the chosen condition.
 
Loads benchmark tasks from 'tasks/benchmark_tasks_v2.csv', instantiates a
smolagents ToolCallingAgent configured for one of three conditions
(no_rag, rag_opt, rag_forced), executes each task, and writes a detailed
log entry to 'logs/run_<run_id>_<condition>.jsonl'.
 
Each log entry records the task, the prompt sent to the agent, the agent's
final answer, every tool call the agent made (including the queries it
issued), and any error encountered.
 
Usage:
    # Run all three conditions on the full benchmark
    python agent_runner.py --condition all --run-id full_run_01
 
    # Run a single condition
    python agent_runner.py --condition rag_opt
 
    # Run a specific task subset
    python agent_runner.py --condition rag_opt --tasks T01 T05 T14

"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from smolagents import ToolCallingAgent, LiteLLMModel, InferenceClientModel
from retrieval_tools import BM25SearchTool, FAISSSearchTool

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM_PROMPT_FILE = Path("system_prompt/system_prompt_v1.txt")
TASKS_FILE = Path("tasks/benchmark_tasks_v2.csv")


def load_system_prompt() -> str:
    with SYSTEM_PROMPT_FILE.open(encoding="utf-8") as f:
        return f.read()


def get_model(provider: str = "anthropic"):
    """
    Return a smolagents model instance for the chosen provider.
    Reads API credentials from environment variables loaded via .env.
    """
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        return LiteLLMModel(model_id="anthropic/claude-haiku-4-5", api_key=api_key)
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        return LiteLLMModel(model_id="gpt-4o-mini", api_key=api_key)
    elif provider == "hf":
        hf_token = os.environ.get("HF_TOKEN")
        return InferenceClientModel(
            model_id="Qwen/Qwen2.5-72B-Instruct", token=hf_token
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def build_agent(condition: str, model) -> ToolCallingAgent:
    """
    Build a ToolCallingAgent for the given experimental condition.
 
    no_rag      : agent is initialised with no retrieval tools
    rag_opt     : both BM25 and FAISS tools are registered
    rag_forced  : both tools are registered (the BM25 instruction is added at the prompt level in build_prompt, not here)
    """
    tools = [] if condition == "no_rag" else [BM25SearchTool(), FAISSSearchTool()]
    agent = ToolCallingAgent(tools=tools, model=model, max_steps=5)
    agent.prompt_templates["system_prompt"] = load_system_prompt()
    return agent


def build_prompt(task: dict, condition: str) -> str:
    """
    Build the user prompt for a single task under a given condition.
 
    Combines the task question with the optional prompt excerpt (for Groups B and D). 
    Under the rag_forced condition, prepends an explicit instruction to call bm25_search before answering.
    """
    question = task["question"]
    excerpt = task.get("prompt_text", "").strip()
    if excerpt and excerpt != "[NO EXCERPT — agent must retrieve]":
        prompt = f"{question}\n\n---\n{excerpt}"
    else:
        prompt = question
    if condition == "rag_forced":
        prompt = (
            "Before answering, you must call bm25_search with a relevant query. "
            "Then use the retrieved results to answer:\n\n" + prompt
        )
    return prompt


def extract_tool_calls(agent: ToolCallingAgent, debug: bool = False) -> list[dict]:
    """
    Extract tool calls from the agent's memory after a run.
 
    smolagents 1.24 stores execution steps in 'agent.memory.steps'. 
    Each ActionStep has a 'tool_calls' list of ToolCall objects with 'name' and 'arguments' attributes. 
    The 'final_answer' call (which submits the agent's response) is excluded because it is not a retrieval call.
    """
    tool_calls = []
    try:
        steps = agent.memory.steps
    except AttributeError:
        return tool_calls

    if debug:
        print(f"    [debug] memory has {len(steps)} steps")
        for i, step in enumerate(steps):
            attrs = [a for a in dir(step) if not a.startswith("_")]
            print(f"    [debug] step[{i}] type={type(step).__name__} | attrs={attrs}")
            if hasattr(step, "tool_calls"):
                print(f"    [debug]   tool_calls={step.tool_calls}")

    for step in steps:
        if hasattr(step, "tool_calls") and step.tool_calls:
            for tc in step.tool_calls:
                name = getattr(tc, "name", "") or ""
                args = getattr(tc, "arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                query = args.get("query", "") if isinstance(args, dict) else ""
                # Exclude final_answer cause its not a retrieval call
                if name and name != "final_answer":
                    tool_calls.append({"tool": name, "query": query})

    return tool_calls


def run_task(
    agent: ToolCallingAgent,
    task: dict,
    condition: str,
    run_id: str,
    debug: bool = False,
) -> dict:
    """Execute a single task and return a complete log record."""
    prompt = build_prompt(task, condition)
    try:
        answer = agent.run(prompt)
        error = None
    except Exception as e:
        answer = ""
        error = str(e)

    tool_calls = extract_tool_calls(agent, debug=debug)
    tools_used = list({tc["tool"] for tc in tool_calls if tc["tool"]})
    retrieval_used = any(t in ["bm25_search", "faiss_search"] for t in tools_used)

    return {
        "run_id": run_id, "task_id": task["task_id"], "condition": condition,
        "group": task.get("group", ""), "gold_retrieval_required": task.get("retrieval_required", ""),
        "gold_best_tool": task.get("best_tool", ""), "prompt": prompt, "answer": str(answer),
        "retrieval_used": retrieval_used, "tools_used": tools_used, "tool_calls": tool_calls,
        "num_retrieval_calls": len(tool_calls), "error": error,
        "retrieval_decision_score": None, "tool_correct": None, "correctness_score": None,
        "grounding_score": None, "hallucination_flag": None, "notes": "",
    }


def load_tasks(path: Path) -> list[dict]:
    """Load benchmark tasks from a CSV file."""
    import csv
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_condition(condition, tasks, model, run_id, task_ids=None, debug=False):
    agent = build_agent(condition, model)
    if task_ids:
        tasks = [t for t in tasks if t["task_id"] in task_ids]
    logs = []
    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task['task_id']} — {condition}")
        log = run_task(agent, task, condition, run_id, debug=debug)
        logs.append(log)
        print(f"    retrieval_used: {log['retrieval_used']} | tools: {log['tools_used']} | calls: {log['num_retrieval_calls']}")
        print(f"    error: {log['error']}")
        print(f"    answer: {str(log['answer'])[:200]}\n")
    return logs


def save_logs(logs, condition, run_id):
    """Write run logs to a JSONL file in the logs directory."""
    filename = LOGS_DIR / f"run_{run_id}_{condition}.jsonl"
    with filename.open("w", encoding="utf-8") as f:
        for entry in logs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(logs)} entries to {filename}")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["no_rag", "rag_opt", "rag_forced", "all"], default="rag_opt")
    parser.add_argument("--provider", choices=["anthropic", "openai", "hf"], default="anthropic")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(TASKS_FILE)
    model = get_model(args.provider)
    conditions = ["no_rag", "rag_opt", "rag_forced"] if args.condition == "all" else [args.condition]

    for condition in conditions:
        print(f"\n{'='*50}\nCondition: {condition}\n{'='*50}\n")
        logs = run_condition(condition, tasks, model, args.run_id, args.tasks, debug=args.debug)
        save_logs(logs, condition, args.run_id)
    print("\nDone.")
