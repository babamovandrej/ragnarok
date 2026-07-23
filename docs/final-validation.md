# Final Validation

All commands below were executed in this repository's `.venv` (Python 3.11.14) after the
last commit of this session. Output is real, captured directly from the commands; nothing
here is fabricated or estimated.

## Commands executed and results

### Lint

```
$ python -m ruff check .
All checks passed!
```

### Format

```
$ python -m ruff format --check .
96 files already formatted
```

### Type check

```
$ python -m mypy
Success: no issues found in 48 source files
```

`[tool.mypy]` in `pyproject.toml` runs with `strict = true` across `src`, `scripts`, and
`app.py`. Two narrowly-scoped `# type: ignore` comments remain in the codebase (both with an
inline reason), and one file-scoped override (`src/generation/zerogpu_backend.py`, for
`transformers`/`spaces` APIs that ship no type stubs) — see
[`refactoring-decisions.md`](refactoring-decisions.md) Task 3.

### Tests, with coverage

```
$ python -m pytest -q --cov --cov-report=term-missing
...
316 passed, 5 deselected, 630 warnings in ~20-26s

---------- coverage: platform darwin, python 3.11.14-final-0 ----------
Name                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
app.py                                  47      4    91%   39-40, 58, 71
scripts/ask.py                          65      1    98%   136
scripts/build_index.py                 222      7    97%   210-211, 274, 287, 469-470, 480
scripts/compare_retrieval.py            45      1    98%   83
scripts/push_index_to_hub.py            42      1    98%   73
scripts/run_eval.py                     59      1    98%   109
src/ingestion/chunker.py               245     22    91%   (per-language edge cases)
src/ingestion/embedder.py               20     11    45%   (SentenceTransformer load path — needs a real model)
src/eval/ragas_runner.py                28      4    86%   59-65 (async agenerate_text path)
src/utils/hf_dataset_io.py              34      2    94%   34-35
src/ingestion/clone.py                 120      2    98%   162-163 (cleanup-of-cleanup OSError path)
src/ingestion/indexer.py                97      4    96%   114-119 (BaseException rollback paths)
... (every other module 100%)
------------------------------------------------------------------
TOTAL                                 1656     60    96%
```

5 tests are deselected by default (`addopts = "-m 'not integration'"` in `pyproject.toml`);
these are the real-network/real-model integration tests, intentionally excluded from the
default run (see "Explicitly not run" below).

The 4% of statements not covered are concentrated in: `src/ingestion/embedder.py` (the
`SentenceTransformer` construction path, which needs a real model download to exercise),
a handful of `BaseException` rollback branches in `src/ingestion/indexer.py`/`clone.py`
(deliberately hard-to-trigger filesystem-race paths), the async `agenerate_text` path in
`src/eval/ragas_runner.py`, and a few CLI argument-error branches in `scripts/*.py`.

### Packaging

```
$ pip install -e ".[dev]" --dry-run
...
Would install ragnarok-0.1.0
```

Resolves cleanly against the pinned versions in `requirements.txt` with no conflicts.

### CLI entry points

```
$ python scripts/build_index.py --help    # exit 0
$ python scripts/ask.py --help            # exit 0
$ python scripts/compare_retrieval.py --help  # exit 0
$ python scripts/run_eval.py --help       # exit 0
$ python scripts/push_index_to_hub.py --help  # exit 0
```

All five CLI scripts parse arguments and print help text successfully.

### Application startup

```
$ python -c "import app; print(type(app.demo))"
<class 'gradio.blocks.Blocks'>
```

Confirms `app.py`'s module-level Gradio `Blocks` app builds successfully, including both
tabs (`build_compare_view`/`build_ingest_view` rendered), the `gr.themes.Soft()` theme, and
`_demo_repo_context()` (which gracefully returns `None` in this environment, since no demo
index has been built here — exercised directly, see below).

### Manual UI logic verification

Since no real index/models exist in this environment (see "Explicitly not run" below), the
UI's rendering logic was exercised directly against fake `PipelineResult`s rather than
through a live browser session:

```
$ python -c "
from src.ui.compare_view import render_comparison
from src.baseline.schemas import PipelineResult
def fake_compare(q):
    return (
        PipelineResult(pipeline='naive', answer='naive answer', citations=[], hop_path=[]),
        PipelineResult(pipeline='ragnarok', answer='ragnarok answer with a.py#L1-L2 cited', citations=[], hop_path=[]),
    )
print(render_comparison(fake_compare, ''))
print(render_comparison(fake_compare, 'a real question'))
"
('_Enter a question above, then press Ask._', '_Enter a question above, then press Ask._')
('naive answer', 'ragnarok answer with a.py#L1-L2 cited\n\n_No citations found_')
```

Confirms the empty-question guard and normal rendering both behave as designed. The
error-path and citation/hop-link rendering are covered by the automated test suite
(`tests/ui/test_compare_view.py`, `tests/ui/test_ingest_view.py`).

## Explicitly not run (descoped per user decision)

The user explicitly chose to skip real-model verification for this pass. Not executed:

- A real `scripts/run_eval.py` run (RAGAS scoring against a live LLM backend) — needs a
  built index and real embedding/rerank/generation inference.
- A live `python app.py` browser click-through with real models (`Embedder`, `Reranker`,
  `CPUBackend` all need multi-GB model downloads and real CPU inference).
- Dispatching `.github/workflows/reindex.yml` (the incremental-reindex GitHub Action) —
  also requires `HF_TOKEN`/`HF_DATASET_REPO`, which per `docs/roadmap.md` were never
  configured for this project.
- Deploying to a live HF Space.

These match the pre-existing, documented status in `README.md`'s "Status" section
(phases 3–9 code-complete, real-model verification runs pending) and were not something this
refactoring pass was asked to complete — see the plan's Context section for the explicit
scope decision.

## Remaining known limitations

- Compare tab still always targets the fixed demo repo (`encode/starlette`); the Ingest tab's
  result is not wired into Compare's repo selection. Flagged during Task 9 as a larger,
  separately-confirmable behavioral change; not implemented.
- Coverage has no enforced minimum in CI (reported, not gated) — see Task 11's decision
  record for why.
- The `gr.Blocks(..., theme=...)` constructor argument is deprecated as of the pinned Gradio
  version (5.50.0) in favor of passing `theme=` to `.launch()`; the replacement API isn't
  available yet on this pinned version (confirmed via `inspect.signature(gr.Blocks.launch)`),
  so the deprecation warning is expected and harmless until Gradio is upgraded.
