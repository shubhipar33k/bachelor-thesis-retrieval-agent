# Thesis: Evaluating Retrieval Decisions in Tool-Calling LLM Agents

This repository contains the code, corpus, benchmark, and experimental logs for the bachelor thesis *Evaluating Retrieval Decisions in Multi-Step LLM Agents* by Shubhi Pareek, submitted to the Department of Computational Linguistics, University of Zurich (June 2026).

## Overview

The project investigates whether a general instruction-tuned LLM agent can reliably decide when to retrieve information, which retrieval tool to use, and when to abstain from retrieval. A tool-calling agent is implemented using the [smolagents](https://github.com/huggingface/smolagents) framework with two retrieval tools (BM25 and FAISS) indexed over a corpus of 11 University of Zurich study regulation documents. The agent is evaluated on a 25-task benchmark spanning four task families under three experimental conditions.

## Repository Structure

```
bachelor-thesis-retrieval-agent/
├── corpus_v1/              # 11 source PDF documents (study regulations)
├── data/                   # Processed chunks and retrieval indices
│   ├── chunks_v1.jsonl     # 46 chunks extracted from the corpus
│   ├── bm25_index.pkl      # BM25 index built from chunks
│   ├── faiss_index.pkl     # FAISS dense retrieval index
│   ├── llm_judge_results.csv   # Per-task judge scores
│   └── llm_judge_results.jsonl # Same data in JSONL format
├── logs/                   # Per-condition run logs (JSONL)
├── scripts/                # All Python scripts
│   ├── preprocess_and_chunk.py  # PDF extraction and chunking
│   ├── build_bm25.py            # Build BM25 index
│   ├── build_faiss.py           # Build FAISS index
│   ├── retrieval_tools.py       # BM25 and FAISS tool implementations
│   ├── agent_runner.py          # Run the agent across conditions
│   └── llm_judge.py             # Evaluate runs with LLM-as-judge
├── system_prompt/          # Agent system prompt
├── tasks/                  # Benchmark task definitions (CSV)
├── requirements.txt
└── README.md
```

## Setup

### Requirements

- Python 3.10 or later
- An Anthropic API key (for the agent and the judge)

### Installation

```bash
git clone https://github.com/shubhipar33k/bachelor-thesis-retrieval-agent.git
cd bachelor-thesis-retrieval-agent
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the repository root containing your Anthropic API key:

```
ANTHROPIC_API_KEY=your_key_here
```

The `.env` file is excluded from version control via `.gitignore`.

## Running the Pipeline

The pipeline runs in four stages. The first three build the corpus and indices and only need to be run once.

### 1. Build the corpus chunks

```bash
python scripts/preprocess_and_chunk.py
```

Extracts text from all PDFs in `corpus_v1/`, cleans the text, and splits it into 300-word chunks with 50-word overlap. Output: `data/chunks_v1.jsonl`.

### 2. Build the BM25 index

```bash
python scripts/build_bm25.py
```

Tokenises the chunks and builds an Okapi BM25 index. Output: `data/bm25_index.pkl`.

### 3. Build the FAISS index

```bash
python scripts/build_faiss.py
```

Encodes the chunks using `paraphrase-multilingual-MiniLM-L12-v2` and builds a FAISS inner-product index. Output: `data/faiss_index.pkl`.

### 4. Run the agent

To run all three experimental conditions on the full benchmark:

```bash
python scripts/agent_runner.py --condition all --run-id full_run_01
```

To run a single condition:

```bash
python scripts/agent_runner.py --condition rag_opt
python scripts/agent_runner.py --condition rag_forced
python scripts/agent_runner.py --condition no_rag
```

Output: per-condition log files in `logs/`.

### 5. Evaluate with LLM-as-judge

```bash
python scripts/llm_judge.py
```

Scores each agent response against the gold answer using Claude Sonnet 4.5 as judge. Output: `data/llm_judge_results.csv` and `data/llm_judge_results.jsonl`.

## Models Used

- **Agent**: Claude Haiku 4.5 (`claude-haiku-4-5`)
- **Judge**: Claude Sonnet 4.5 (`claude-sonnet-4-5`)
- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)

Total API cost for the full experimental run (75 agent runs + 75 judge calls) was under 2 USD.

## Benchmark

The benchmark contains 25 tasks distributed across four families:

- **Group A** (8 tasks): Retrieval necessary
- **Group B** (8 tasks): Retrieval unnecessary (excerpt provided)
- **Group C** (3 tasks): Ambiguous
- **Group D** (6 tasks): Adversarial (explicit abstain instruction)

Each task is annotated with a gold answer, a gold retrieval label, a gold tool label, and a gold chunk source. The full benchmark is in `tasks/benchmark_tasks_v2.csv`.

## Corpus

The corpus consists of 11 publicly available PDF documents covering the Bachelor programme in Computational Linguistics and Language Technology at the University of Zurich, including study regulations, assessment rules, plagiarism guidelines, and guidelines on the use of text generation models. All documents are in German or English. The corpus is frozen and was not modified after the benchmark tasks were written.

## License

The code in this repository is released under the MIT License. The corpus documents are public University of Zurich materials and remain the property of their original authors.

## Citation

If you do reference this work, cite like this:

```
Pareek, S. (2026). Evaluating Retrieval Decisions in Multi-Step LLM Agents.
Bachelor thesis, Department of Computational Linguistics, University of Zurich.
```
