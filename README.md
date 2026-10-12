# RAGnarok

Ask questions about a GitHub codebase and get answers that cite the exact lines they came
from, with every citation checked against the index. Each question is also answered by a
naive RAG baseline (dense retrieval only), so the two can be compared side by side.
Everything runs locally on open-weight models.

## Results

Measured on [Starlette](https://github.com/encode/starlette) at commit `5174d4c` (615 code
chunks) with [35 hand-labeled questions](data/qa_starlette.jsonl): 20 single-hop, 10
multi-hop, and 5 about features Starlette doesn't have.

### Retrieval

| Retrieval setup | Single-hop hit@1 | Single-hop MRR | Multi-hop hit@1 | Multi-hop MRR | Multi-hop file recall@10 |
|---|---|---|---|---|---|
| Dense only (the naive baseline) | 0.90 | 0.95 | 0.90 | 0.95 | **0.80** |
| BM25 only | 0.45 | 0.62 | 0.80 | 0.90 | 0.60 |
| Hybrid (dense + BM25, fused) | 0.80 | 0.90 | 0.80 | 0.90 | 0.72 |
| **Hybrid + rerank (RAGnarok)** | **0.95** | **0.98** | **1.00** | **1.00** | 0.72 |

- *hit@1*: the file holding the answer is ranked first.
- *MRR*: mean reciprocal rank of the first correct file.
- *File recall@10*: the share of the files a multi-hop answer needs that appear in the top 10.

One question is worth 5 points of single-hop hit@1 and 10 points of multi-hop hit@1.

### Questions the repo can't answer

The reranker's best relevance score was at most 0.50 on all five questions about missing
features and at least 2.73 on all 30 real ones. Below a cutoff of 1.0, RAGnarok answers "not
covered by this repo" without calling the LLM. The baseline answers every question.

### End-to-end answers

Five of the questions, run through both pipelines and judged by hand:

| Question | Naive baseline | RAGnarok | RAGnarok's citation | Hops |
|---|---|---|---|---|
| What Response subclass sends a redirect to another URL? | correct | correct | verified | 0 |
| What decorator restricts an endpoint to users with specific auth scopes? | correct | correct | verified | 0 |
| How does `@requires` redirect an unauthenticated request? | correct | correct | verified | 0 |
| How does `ExceptionMiddleware` turn an `HTTPException` into a response? | correct | correct | none given | 1 |
| What ORM does Starlette use? | answers from general knowledge | refuses | n/a | 0 |

## Setup

You need Python 3.11, `git`, and about 7 GB of free memory for the three models.

```bash
git clone https://github.com/babamovandrej/ragnarok.git
cd ragnarok
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

The models download from the Hugging Face Hub on first use, about 4 GB in total:
Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B and Qwen2.5-Coder-3B-Instruct (the last one is
licensed for non-commercial use). On macOS, `llama-cpp-python` builds from source, which
takes a few minutes.

## Run

Index a repository. This one is Starlette at the commit the results above were measured on:

```bash
python -m scripts.build_index https://github.com/encode/starlette \
    --revision 5174d4c8358a6f06aa8056bafd14c2272dab8dd1
```

Ask a question from the command line:

```bash
python -m scripts.ask .ragnarok_cache/encode/starlette \
    "How does the @requires decorator redirect an unauthenticated request?" --agent
```

Or start the web app and open http://127.0.0.1:7860. It shows the baseline and RAGnarok side
by side, and its "Ingest a repo" tab indexes any public GitHub repository:

```bash
python app.py
```

To regenerate the results above:

```bash
python -m scripts.eval_retrieval .ragnarok_cache/encode/starlette
python -m scripts.sample_answers .ragnarok_cache/encode/starlette
```

The embedding and reranking models use a GPU when one is available. To keep them on the
CPU, set `RAGNAROK_DEVICE=cpu`, for example `RAGNAROK_DEVICE=cpu python app.py`.
