"""Upload validation and analyze acceptance tests."""

import io


def test_analyze_accepts_png(client, png_bytes):
    res = client.post(
        "/api/analyze",
        files={"file": ("sample.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["case_id"].startswith("DS-")
    assert body["upload"]["width"] == 320
    assert body["upload"]["height"] == 200
    auth = body["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    assert body["assessment"]["decision"] in {"PENDING", "INCONCLUSIVE"}
    assert "image_quality" in body
    assert "document_type" in body


def test_analyze_rejects_empty(client):
    res = client.post(
        "/api/analyze",
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert detail["code"] == "EMPTY_FILE"


def test_analyze_rejects_bad_extension(client, png_bytes):
    res = client.post(
        "/api/analyze",
        files={"file": ("payload.exe", io.BytesIO(png_bytes), "application/octet-stream")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "BAD_EXTENSION"


def test_analyze_rejects_corrupt(client):
    res = client.post(
        "/api/analyze",
        files={"file": ("bad.png", io.BytesIO(b"not-an-image"), "image/png")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["code"] in {"CORRUPT_IMAGE", "BAD_SIGNATURE"}


def test_case_roundtrip(client, png_bytes):
    created = client.post(
        "/api/analyze",
        files={"file": ("sample.png", io.BytesIO(png_bytes), "image/png")},
    ).json()
    case_id = created["case_id"]
    res = client.get(f"/api/case/{case_id}")
    assert res.status_code == 200
    assert res.json()["case_id"] == case_id
    assert res.json()["decision"] in {"PENDING", "INCONCLUSIVE"}
    assert res.json()["document_type"] in {
        "PAN",
        "AADHAAR",
        "PASSPORT",
        "DRIVING_LICENCE",
        "VISA",
        "NATIONAL_ID",
        "PERMIT",
        "OTHER",
        "UNKNOWN",
    }


def test_reverse_search_unavailable(client):
    res = client.post("/api/reverse-search")
    assert res.status_code == 200
    assert res.json()["status"] == "SEARCH_UNAVAILABLE"


def test_face_compare_without_file_rejected(client):
    res = client.post("/api/face/compare")
    assert res.status_code in {400, 422}
