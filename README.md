# RAGnarok

A RAG assistant for understanding codebases. Paste a public GitHub repo URL,
ask a natural-language question, and get an answer grounded in the actual
code — with verified `file#Lstart-Lend` citations and multi-hop tracing
across files.

This project is under active development. See [`docs/roadmap.md`](docs/roadmap.md)
for the phase-by-phase build plan, [`docs/architecture.md`](docs/architecture.md)
for the current design, and [`DECISIONS.md`](DECISIONS.md) for the reasoning
behind every non-obvious choice.

Architecture diagram, demo, and evaluation numbers land in Phase 10 — see
[`docs/phases/phase-10-polish.md`](docs/phases/phase-10-polish.md).

## Status

Phase 1 (scaffold) in progress. Not yet deployed.

## Local development

```bash
pip install -e .
pip install -r requirements.txt
pytest -q
python app.py
```
