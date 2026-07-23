# Starlette hand-labeled eval set

`qa_starlette.jsonl` — 35 items against `encode/starlette`, hand-labeled
against the real repo at commit `5174d4c8358a6f06aa8056bafd14c2272dab8dd1`
(cloned read-only to verify every fact below; no ML model involved).

## Schema (`src/eval/qa_dataset.py`'s `QAItem`)

| Field | Meaning |
|---|---|
| `id` | Stable identifier, `<category>_<NN>`. |
| `question` | The natural-language question asked of RAGnarok. |
| `category` | `single_hop`, `multi_hop`, or `adversarial`. |
| `expected_files` | Repo-relative file(s) a correct, grounded answer should cite. Empty for adversarial items — nothing in the repo should be cited. |
| `grounding_notes` | Human-verified explanation of why the expected files/trace are correct, so a reviewer can check without re-deriving it. |
| `expect_idk` | `true` only for adversarial items: a correct answer is an explicit "I don't know," not a guess. |

## Composition (35 items)

- **20 single-hop** — "where is X defined" / "what does X do" questions
  answerable from one file (e.g. `AuthenticationMiddleware`,  `Route`,
  `Response` subclasses, individual middleware classes).
- **10 multi-hop** — questions whose correct answer requires tracing
  across ≥2 files (e.g. the auth path from `middleware/authentication.py`
  through `authentication.py` to `requests.py`'s `Request.auth`/`.user`).
- **5 adversarial** — real-sounding questions about capabilities Starlette
  genuinely does not have (GraphQL, an ORM, OAuth2 provider integrations,
  rate limiting, an admin dashboard). A correct RAGnarok answer says "I
  don't know" rather than fabricating an answer from unrelated context.

## How each item was labeled

Every `expected_files`/`grounding_notes` pair was checked directly against
a shallow clone of `encode/starlette` at the commit above — reading the
actual class/function definitions (not recalled from memory) — before
being added here. Adversarial items were checked the same way: confirming
the claimed capability doesn't appear anywhere in the indexed source.

## Using this set

`scripts/run_eval.py --pipeline ragnarok` runs every item through the full
pipeline and scores it with RAGAS (faithfulness, context precision/
recall) using our own model as judge — see `src/eval/ragas_runner.py`.
`scripts/run_eval.py --ablation` runs the same 35 items through naive /
hybrid+rerank / full multi-hop and reports one 3-row table — see
`src/eval/ablation.py`. `src/eval/metrics.py`'s `idk_rate` checks what
fraction of the 5 adversarial items produce an explicit "I don't know"
(target: ≥80%, i.e. at least 4 of 5).

Neither script has been run against a real model in this repo yet (see
`DECISIONS.md`, Phase 7 entry) — running them requires a built starlette
index and real embedding/rerank/generation inference.
