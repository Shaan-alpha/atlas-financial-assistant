from atlas.tools.result import err, ok


def test_ok_carries_source_and_timestamp():
    r = ok({"price": 101.5}, source="yfinance")

    assert r["ok"] is True
    assert r["data"]["price"] == 101.5
    assert r["source"] == "yfinance"
    assert r["as_of"].endswith("Z")


def test_ok_accepts_explicit_as_of():
    r = ok({"x": 1}, source="sec-edgar", as_of="2026-08-07T00:00:00Z")
    assert r["as_of"] == "2026-08-07T00:00:00Z"


def test_err_shape_is_reasonable_for_a_model_to_read():
    r = err("no_such_symbol", "No listed security matches 'XYZQ'.")

    assert r["ok"] is False
    assert r["error"] == "no_such_symbol"
    assert "XYZQ" in r["message"]
    assert "data" not in r
