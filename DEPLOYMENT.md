# OMNIVISTA AI — Vercel Deployment Guide (Multi-Service)

This guide deploys OMNIVISTA AI as **two independent Vercel projects**:

| Project | Purpose | Runtime | Directory |
|---|---|---|---|
| `omnivista-api` | FastAPI backend (`/api/*`, ingestion, retrieval, generation) | Python 3.11 (serverless) | `backend/` |
| `omnivista-web` | Next.js frontend (document library, split-screen viewer + chat) | Node.js (Next.js) | `frontend/` |

> **Why two projects?** The backend and frontend have different runtimes and
> scaling profiles. Splitting them lets us (a) scale the API independently,
> (b) avoid mixing Node and Python in one function bundle, and (c) set
> different region / timeout / memory settings.

---

## 0. Architecture

```
                    VERCEL
         ┌──────────────────────────────┐
         │  omnivista-web (Next.js)      │
         │   · library · viewer · chat   │
         └───────────────┬──────────────┘
                         │ HTTPS (server-side fetch)
         ┌───────────────▼──────────────┐
         │  omnivista-api (FastAPI)      │
         │   /api/documents,/api/chat…   │
         └───────────────┬──────────────┘
             ┌───────────┼───────────────┐
             ▼           ▼               ▼
      PostgreSQL      Vector Store     AI Services
      (Neon/Supabase)  (Pinecone)     (LLM/VLM/Embeddings)
             │
             ▼
      Object Storage (Vercel Blob / S3 / Supabase Storage)
             │
             ▼
      Background Ingestion Worker  (NOT a serverless function)
```

**Never run PostgreSQL or a vector database inside Vercel.** They are
externally managed services reached over the network.

---

## 1. Prerequisites

- A Vercel account + the CLI (`npm i -g vercel`).
- A PostgreSQL provider: **Neon** or **Supabase** (attach a connection string).
- A vector store: **Pinecone** (or Chroma for self-host).
- Object storage: **Vercel Blob**, **S3** (AWS/Supabase), or **Supabase Storage**.
- AI provider keys (OpenAI / Google / Anthropic) — optional in mock mode.

---

## 2. Backend: `omnivista-api`

### 2.1 Vercel config (`backend/vercel.json`)

```json
{
  "name": "omnivista-api",
  "version": 2,
  "regions": ["bom1"],
  "builds": [{ "src": "api/index.py", "use": "@vercel/python", "config": { "maxDuration": 60, "memory": 1024 } }],
  "functions": { "api/index.py": { "maxDuration": 60, "memory": 1024, "runtime": "python3.11" } },
  "routes": [{ "src": "/(.*)", "dest": "api/index.py" }]
}
```

- `api/index.py` imports the ASGI app from `app.main` and exposes `handler`.
- The Python bundle works in **mock mode** (zero external services).

### 2.2 Environment variables (`omnivista-api`)

Set these in the Vercel dashboard under **Settings → Environment Variables**.

```env
ENVIRONMENT=production
MOCK_MODE=false

DATABASE_URL=postgres://…          # Neon/Supabase pooled connection
OPENAI_API_KEY=…
GOOGLE_API_KEY=…
ANTHROPIC_API_KEY=…

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=…

VECTOR_DB_PROVIDER=pinecone
PINECONE_API_KEY=…
PINECONE_INDEX=omnivista

STORAGE_PROVIDER=vercel-blob
BLOB_READ_WRITE_TOKEN=…

CORS_ORIGINS=https://<frontend-domain>
JWT_SECRET=…
```

### 2.3 Deploy

```bash
cd backend
vercel login
vercel link --project omnivista-api --yes
vercel --prod
```

To split the monorepo, set the project **Root Directory** to `backend`.

> Serverless **function size / cold-start**: the parsing stack (pdfplumber,
> reportlab, matplotlib, numpy) is heavy. Keep it behind the mock/PDF path; for
> very large documents, shift ingestion to the background worker (section 5).
> You may also strip matplotlib/reportlab from the request path by keeping
> generated demo assets pre-built.

---

## 3. Frontend: `omnivista-web`

A Next.js (App Router) + TypeScript + Tailwind CSS app.

Scaffold:

```text
frontend/
├── app/
│   ├── layout.tsx
│   ├── page.tsx            # Dashboard
│   ├── documents/page.tsx
│   ├── chat/page.tsx       # split-screen viewer + chat
│   ├── analytics/page.tsx
│   └── globals.css
├── components/
├── features/  {documents, chat, viewer, analytics, evaluation}
├── lib/api.ts              # typed API client (uses NEXT_PUBLIC_API_URL)
├── hooks/
└── package.json
```

### 3.1 API client

Never hardcode production URLs. Read `NEXT_PUBLIC_API_URL` and build a small
typed client:

```ts
// lib/api.ts
export const API_URL = process.env.NEXT_PUBLIC_API_URL!;
export const fetcher = <T>(path: string, init?: RequestInit) =>
  fetch(`${API_URL}${path}`, init).then((r) => r.json() as Promise<T>);
```

For server-side rendering / `route handlers`, use `fetch` with the URL from
`process.env.NEXT_PUBLIC_API_URL` (server-reachable), and add `cache: "no-store"`.

### 3.2 `frontend/vercel.json`

```json
{
  "name": "omnivista-web",
  "version": 2,
  "regions": ["bom1"],
  "framework": "nextjs",
  "buildCommand": "next build",
  "outputDirectory": ".next"
}
```

### 3.3 Environment variables (`omnivista-web`)

```env
NEXT_PUBLIC_API_URL=https://omnivista-api.vercel.app/api
NEXT_PUBLIC_APP_URL=https://omnivista-web.vercel.app
```

### 3.4 Deploy

```bash
cd frontend
npm install
vercel link --project omnivista-web --yes
vercel --prod
```

---

## 4. Connecting the two services

1. Deploy `omnivista-api` first and copy its production URL.
2. Set `NEXT_PUBLIC_API_URL` on `omnivista-web` to `https://<api-deploy>/api`.
3. In `omnivista-api`, set `CORS_ORIGINS` to exactly `https://<frontend-domain>`
   (never `*` in production).

Optional: set a custom domain on each project.

---

## 5. Background ingestion (the important part)

**Do not run long document processing inside a request/function.** The upload
endpoint returns `document_id` immediately; a **background worker** does the
parsing → chunking → embedding → indexing.

Recommended pattern:

```text
POST /api/documents            (upload)
   ├─ store object → object storage
   ├─ insert Document (status=UPLOADED)
   └─ enqueue job → managed queue        (Upstash Redis + worker / Inngest / Trigger.dev / SQS + Lambda)
        ▼
   Worker (container or long-running service):
       parse → OCR → layout → vision → table → chunk → embed → index
       update Document status incl. progress
        ▼
   GET /api/documents/{id}/status  (frontend polls / SSE)
```

- In mock mode the queue is `InlineTaskQueue` (runs synchronously). Swap to
  a production queue via `TASK_QUEUE_PROVIDER` without touching the pipeline.
- The pipeline is **idempotent** (chunks keyed by `chunk_id`), so it can be
  resumed/retried.

---

## 6. Database (managed, external)

- PostgreSQL via **Neon** or **Supabase** (pooled connection strings).
- Use **SQLAlchemy 2.0 + Alembic** for migrations and migrations-run on deploy.
- Serverless-safe pooling: SQLAlchemy `AsyncEngine` with `NullPool` or
  `QueuePool` sized for short functions; consider PgBouncer (Neon offers it).
- Multi-region: prefer a region close to your Vercel function region (`bom1`).

---

## 7. Object storage

Raw files **never** live in Postgres. `StorageProvider` supports:

- `local` (dev) — default in mock mode
- `vercel-blob` — `BLOB_READ_WRITE_TOKEN`
- `s3` — S3-compatible (AWS / Supabase / MinIO)

Store the object key (not bytes) in Postgres.

---

## 8. Vector store

Prefer **Pinecone** for production. The `VectorStore` abstraction keeps the
retrieval code independent of the backend:

```text
VectorStore.upsert(chunk_id, embedding, metadata)
VectorStore.search(query_embedding, top_k, filters)
VectorStore.delete_document(document_id)
```

Every vector carries `document_id, page_number, chunk_id, content_type,
section, bbox, source_type` for precise citations.

---

## 9. CORS & security

- `CORS_ORIGINS` is a comma-separated allow-list; enforce exact production
  origins (`https://omnivista-web…`), never `*`.
- Validate file **extension, MIME, and size** at upload (see
  `MAX_UPLOAD_BYTES`); sanitize filenames; never execute uploads.
- Treat uploaded documents as **untrusted input**: retrieved content is wrapped
  in a separate, clearly-marked `[EVIDENCE]` block and the generation prompt
  instructs the model to treat it as data, not instructions (prompt-injection
  defense). The mock LLM is extractive and evidence-gated by construction.
- Add rate limiting (e.g. `slowapi` or Vercel firewall) and JWT auth in the
  auth-ready layer when enabled.
- Never log secrets; use structured logging with a redaction hook.

---

## 10. CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs lint → type-check → backend tests → build →
integration tests on every push and on PRs.

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  backend:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: backend } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python -m pytest -q
      - run: python -m ruff check . || true
```

Production deploys are triggered by the Vercel GitHub integration per project
(or via `vercel --prod` in a deploy job).

---

## 11. Verifying the deployment

```bash
# API
curl https://omnivista-api.vercel.app/health
curl -X POST https://omnivista-api.vercel.app/api/demo/seed?kind=pdf
curl https://omnivista-api.vercel.app/api/documents/<id>/status

# Chat
curl -X POST https://omnivista-api.vercel.app/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What does the architecture diagram show?","document_ids":["<id>"]}'
```

Then open the frontend, generate the demo, and ask the acceptance-test questions
(diagram, chart, table, cross-modal).

---

## 12. Phases

- **MVP (this repo, mock mode):** PDF upload → text/table/image extraction →
  semantic multimodal chunking → embeddings → vector search → grounded answer
  → page/region citations → split-screen viewer + chat → processing status.
- **Phase 2:** real LLM/VLM/embedding providers, OCR, hybrid search, reranking,
  multi-document chat, analytics, object storage.
- **Phase 3:** agentic RAG loop, evidence validation, knowledge graph,
  RAG evaluation, model A/B testing, prompt-injection hardening.
- **Phase 4 (research mode):** text-vs-multimodal experiments, metrics and
  comparison reports.

Each provider is behind an interface (`LLMProvider`, `VisionProvider`,
`EmbeddingProvider`, `VectorStore`, `StorageProvider`, `TaskQueue`,
`RerankerProvider`), so moving from mock → production is configuration-only.
