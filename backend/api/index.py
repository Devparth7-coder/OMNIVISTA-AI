"""Vercel serverless entrypoint for the FastAPI app.

Vercel's Python runtime imports this module (``api/index.py``) and calls its
``app`` attribute. ``from app.main import app`` pulls the full FastAPI
application; see ``vercel.json`` for the function configuration.

Note for production: long-running ingestion must NOT run inside a serverless
function. Move the heavy ingestion worker to a background service (see the
repo's DEPLOYMENT.md) and trigger it from the API via a managed queue.
"""
import os
import sys

# Make the project root importable when run under Vercel's function sandbox.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402

# Expose the ASGI app (Vercel expects `app` or a `handler`).
handler = app
