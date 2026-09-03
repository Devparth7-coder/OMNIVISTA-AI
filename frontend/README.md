# omnivista-web (frontend)

Next.js (App Router, `/app`), TypeScript, Tailwind CSS, shadcn/ui, Recharts,
React PDF (`react-pdf`) / PDF.js for the viewer.

> **Status:** The runnable MVP client is served by the FastAPI backend
> (`backend/web/index.html`, opened at the API root) so the whole flow works in
> mock mode with zero build. This directory is the production Next.js frontend
> scaffold. For a full build, scaffold it with:

```bash
npx create-next-app@latest frontend --ts --tailwind --app --eslint
```

Then add the routes and feature folders described in `../DEPLOYMENT.md`:

```text
app/
  layout.tsx
  page.tsx              # Dashboard
  documents/page.tsx    # Document library + upload
  chat/page.tsx         # Split-screen viewer + chat
  analytics/page.tsx
  evaluation/page.tsx
components/  # shadcn/ui primitives
features/
  documents/ chat/ viewer/ analytics/ evaluation/
lib/
  api.ts     # typed client reading NEXT_PUBLIC_API_URL
hooks/
```

### Run locally

```bash
npm install
cp .env.example .env   # set NEXT_PUBLIC_API_URL=http://localhost:8000/api
npm run dev            # http://localhost:3000
```

### Build

```bash
npm run build
```

### Deploy

```bash
vercel link --project omnivista-web --yes
vercel --prod
```
