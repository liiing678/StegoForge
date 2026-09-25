"""tests/test_multi_carrier.py — End-to-end tests for encode-multi / decode-multi."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from PIL import Image

import stegoforge
from core import framing


@pytest.fixture()
def carriers(tmp_path):
    paths = []
    rng = np.random.default_rng(42)
    for i in range(3):
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        p = tmp_path / f"cover{i}.png"
        Image.fromarray(arr, "RGB").save(p)
        paths.append(p)
    return paths


def _encode(carriers, payload_bytes: bytes, key="testkey", **kw):
    payload_path = carriers[0].parent / "payload.bin"
    payload_path.write_bytes(payload_bytes)
    return stegoforge.op_encode_multi(
        [str(c) for c in carriers], str(payload_path), key,
        kw.get("output_dir"), kw.get("method"), kw.get("depth", 1), False,
    )


def test_multi_roundtrip_out_of_order(carriers):
    payload = os.urandom(3000)  # needs 3 carriers at depth 1
    res = _encode(carriers, payload)
    assert res["chunks"] == 3
    outs = [o["output"] for o in res["outputs"]]
    # Pass parts back shuffled — frames carry their own index.
    dec = stegoforge.op_decode_multi([outs[2], outs[0], outs[1]], "testkey", None, None, False)
    assert Path(dec["output"]).read_bytes() == payload


def test_multi_single_carrier_when_it_fits(carriers):
    res = _encode(carriers[:1], os.urandom(1000))
    assert res["chunks"] == 1
    dec = stegoforge.op_decode_multi([res["outputs"][0]["output"]], "testkey", None, None, False)
    assert dec["payload_size"] == 1000


def test_over_capacity_fails_before_any_write(carriers, tmp_path):
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(os.urandom(100_000))
    before = {p.name for p in tmp_path.iterdir()}
    with pytest.raises(framing.CapacityError, match="Short by"):
        stegoforge.op_encode_multi(
            [str(c) for c in carriers], str(payload_path), "k",
            None, None, 1, False,
        )
    after = {p.name for p in tmp_path.iterdir()}
    assert before == after  # no partial outputs left behind


def test_missing_chunk_error(carriers):
    res = _encode(carriers, os.urandom(3000))
    outs = [o["output"] for o in res["outputs"]]
    with pytest.raises(framing.MissingChunkError, match="Missing 1 of 3"):
        stegoforge.op_decode_multi([outs[0], outs[2]], "testkey", None, None, False)


def test_truncated_part_error(carriers, tmp_path):
    res = _encode(carriers, os.urandom(3000))
    outs = [o["output"] for o in res["outputs"]]
    trunc = tmp_path / "trunc.png"
    data = Path(outs[0]).read_bytes()
    trunc.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="trunc.png"):
        stegoforge.op_decode_multi([str(trunc), outs[1], outs[2]], "testkey", None, None, False)


def test_wrong_key_error(carriers):
    res = _encode(carriers, os.urandom(3000))
    outs = [o["output"] for o in res["outputs"]]
    with pytest.raises(ValueError, match="Decryption failed"):
        stegoforge.op_decode_multi(outs, "wrongkey", None, None, False)


def test_capacity_reports_overhead_aware_estimate(carriers):
    cap = stegoforge.op_capacity(str(carriers[0]), None, 1)
    est = cap["chunked_payload_estimate"]
    assert est["usable_bytes"] == cap["capacity_bytes"] - framing.FRAME_HEADER_SIZE
    assert est["frame_overhead_bytes"] == framing.FRAME_HEADER_SIZE
