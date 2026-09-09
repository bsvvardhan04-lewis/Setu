"""HTTP surface."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from setu.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_system_endpoint(client):
    body = client.get("/api/system").json()
    assert "host" in body and "router" in body
    assert body["offline_only"] is True


def test_languages_endpoint(client):
    body = client.get("/api/languages").json()
    assert "hi" in body["languages"] and "ta" in body["languages"]


def test_form_templates_are_listed(client):
    keys = {t["key"] for t in client.get("/api/forms").json()["templates"]}
    assert {"pension", "scholarship", "grievance"} <= keys


def test_ingest_then_ask_round_trip(client):
    buffer = io.BytesIO()
    Image.new("RGB", (500, 700), "white").save(buffer, format="PNG")
    buffer.seek(0)

    ingest = client.post(
        "/api/ingest",
        files={"files": ("page.png", buffer, "image/png")},
        data={"title": "Test notice"},
    )
    assert ingest.status_code == 200
    document_id = ingest.json()["document_id"]

    answer = client.post(
        "/api/ask", json={"question": "What is the deadline?", "document_id": document_id}
    )
    assert answer.status_code == 200
    assert "answer" in answer.json()


def test_empty_question_is_rejected(client):
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400


def test_form_flow(client):
    started = client.post(
        "/api/forms/start", json={"session_id": "api-1", "template": "pension"}
    )
    assert started.status_code == 200
    assert started.json()["next_prompt"]

    filled = client.post(
        "/api/forms/fill",
        json={"session_id": "api-1", "utterance": "9876543210", "field_key": "mobile"},
    )
    assert filled.json()["field"]["value"] == "9876543210"


def test_unknown_form_template_is_404(client):
    response = client.post(
        "/api/forms/start", json={"session_id": "x", "template": "nope"}
    )
    assert response.status_code == 404
