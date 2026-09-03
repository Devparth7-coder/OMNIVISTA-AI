# OMNIVISTA AI

### Multimodal Retrieval-Augmented Generation & Document Intelligence Platform

OMNIVISTA AI is a **multimodal RAG** platform that understands complex documents
where important information lives in **text, tables, images, charts, and
diagrams**. Users upload a document, ask natural-language questions, and receive
answers that are *grounded in the actual document* with **precise, clickable
citations** that jump to the exact page — and highlight the exact region — of the
source.

This is **not** a basic PDF chatbot. The ingestion stage performs layout
analysis, table extraction, image/chart/diagram understanding, and builds
**semantic multimodal chunks** that are embedded and indexed for hybrid
(vector + lexical) retrieval.

---

## What's in this repo (MVP, mock mode)

A **fully runnable end-to-end MVP** that works with **zero external API keys or
databases**. Every AI/data provider sits behind a clean interface and is backed
by a deterministic mock implementation, so you can:

```
Upload demo PDF
  → parser reads text, tables, images, diagrams (with bounding boxes)
  → layout analysis → multimodal chunking
  → embeddings → vector index
  → ask "What does the architecture diagram show?"
  → retrieval router → hybrid retrieval → rerank
  → grounded, evidence-gated answer with [Page N, Diagram] citation
  → click the citation → viewer highlights the source page/region
```

### The acceptance path (works out of the box)

| Question | Result |
|---|---|
| *What does the architecture diagram show?* | "…client → api gateway → backend → database" — `[Page 1, Diagram]` |
| *Which year had the highest revenue?* | "The highest value is 2024 at 19.0M…" — `[Page 2, Chart]` |
| *Which component connects client→backend?* | "The api gateway sits between client and backend…" — `[Page 1, Diagram]` |
| *What is the revenue in 2025 for North?* | "North reported 16M for 2025" — `[Page 1, Table]` |
| *Does the text match the diagram?* | Cross-modal comparison, cites text + diagram, concludes "consistent" |

---

## Tech stack

- **Backend:** Python 3.11 · FastAPI · pydantic · pdfplumber (layout/bbox) ·
  python-docx / python-pptx · reportlab · matplotlib
- **Providers (pluggable):** `LLMProvider`, `VisionProvider`,
  `EmbeddingProvider`, `VectorStore`, `StorageProvider`, `TaskQueue`,
  `RerankerProvider` — all with mock implementations + swap-in for real ones.
- **Frontend:** a premium dark split-screen client served by the backend
  (document library → upload → processing status → PDF viewer → grounded chat
  → clickable citations → region highlighting). A production Next.js client is
  scaffolded in `frontend/` (see `DEPLOYMENT.md`).
- **Deployment:** Vercel multi-service (`omnivista-api` FastAPI + `omnivista-web`
  Next.js), managed Postgres, Pinecone, Vercel Blob. See **`DEPLOYMENT.md`**.

---

## Repository layout

```text
omnivista/
├── backend/
│   ├── app/
│   │   ├── api/            # routers: documents, chat, search, analytics, demo, health
│   │   ├── ingestion/      # pipeline + vision annotation
│   │   ├── parsers/        # PDF / DOCX / PPTX / text / md + registry
│   │   ├── ocr/            # OCRProvider abstraction
│   │   ├── vision/         # VisionProvider abstraction (image/chart/diagram)
│   │   ├── chunking/       # semantic multimodal chunker
│   │   ├── embeddings/     # EmbeddingProvider abstraction
│   │   ├── vectorstore/    # VectorStore abstraction (+ in-memory)
│   │   ├── retrieval/      # hybrid (vector + BM25) retrieval + router
│   │   ├── reranking/      # RerankerProvider abstraction
│   │   ├── generation/     # LLMProvider abstraction
│   │   ├── evaluation/     # evidence validator + hallucination metrics
│   │   ├── citations/      # citation building
│   │   ├── storage/        # object storage abstraction
│   │   ├── tasks/          # TaskQueue abstraction
│   │   ├── services/       # container, chat, query, context, agent, generation
│   │   ├── demo/           # demo PDF generator
│   │   └── main.py         # FastAPI app + serves the web client
│   ├── web/index.html      # runnable MVP client
│   ├── api/index.py        # Vercel Python entrypoint
│   ├── tests/              # parser / RAG / security / API tests
│   ├── vercel.json
│   └── requirements.txt
├── frontend/               # production Next.js client (scaffold)
├── .github/workflows/ci.yml
├── DEPLOYMENT.md           # Vercel multi-service guide
└── README.md
```

---

## Run it (mock mode — no keys needed)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` → click **Generate demo PDF** → watch it process →
ask one of the suggested questions.

### Tests

```bash
cd backend
python -m pytest -q          # 20 tests: parsers, end-to-end RAG, security, API
```

---

## Design principles

- **Provider abstractions:** swap mock ↔ real (OpenAI/Cohere, Pinecone/Chroma,
  Vercel Blob/S3, managed queues) via env config — never by editing pipeline code.
- **Grounded generation:** the model answers only from retrieved evidence, and
  an **evidence validator** checks groundedness, citation validity, and numeric
  consistency before an answer is trusted.
- **Prompt-injection defense:** retrieved document content is clearly separated
  as untrusted `[EVIDENCE]`; instructions inside documents are treated as data,
  not obeyed.
- **Multimodal chunks:** a chunk can contain a heading + paragraph + table
  summary + image description + caption (+ bbox), so retrieval returns a
  coherent unit of evidence — not a random text slice.

## Roadmap

See `DEPLOYMENT.md` §12 — Phase 2 (real providers, OCR, reranking, multi-doc,
analytics), Phase 3 (agentic RAG, knowledge graph, evaluation, A/B), Phase 4
(research mode).

---

## Security

Uploaded files are validated (extension/MIME/size), filenames sanitized, and
never executed. See `DEPLOYMENT.md` §9 and `backend/tests/test_security.py`.
