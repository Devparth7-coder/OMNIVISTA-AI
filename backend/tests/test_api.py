"""API endpoint validation tests (FastAPI TestClient)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_status_reports_mock_mode():
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["mock_mode"] is True


def test_upload_rejects_invalid_extension():
    r = client.post("/api/documents", files={"file": ("malware.exe", b"x", "application/octet-stream")})
    assert r.status_code == 415


def test_upload_rejects_empty_file():
    r = client.post("/api/documents", files={"file": ("a.pdf", b"", "application/pdf")})
    assert r.status_code == 400


def test_get_missing_document_404():
    r = client.get("/api/documents/nope")
    assert r.status_code == 404


def test_root_serves_web_client():
    r = client.get("/")
    assert r.status_code == 200
    assert "OMNIVISTA" in r.text
    assert "text/html" in r.headers["content-type"]
