"""Health endpoint smoke tests."""


def test_health_ok(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["service"] == "DOCSHIELD AI"
    assert body["components"]["upload_validation"] == "working"
    assert body["components"]["ocr"].startswith("working")
    assert body["components"]["official_verification"] == "unavailable"
    assert body["components"]["reverse_search"] == "unavailable"


def test_api_root(client):
    res = client.get("/api")
    assert res.status_code == 200
    assert res.json()["service"] == "DOCSHIELD AI"
