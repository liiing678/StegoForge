"""tests/test_fragment.py — Multi-carrier fragmentation: capacity planning,
split/join, and clear errors for missing/out-of-order/truncated fragments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core import fragment as frag
from core.fragment import CapacityError, FragmentError

PAYLOAD = bytes(range(256)) * 10  # 2560 bytes


# ── Capacity planning ─────────────────────────────────────────────────────────

def test_plan_accounts_for_header_overhead():
    # 3 carriers × 133 raw → 100 usable each after the 33-byte header.
    sizes = frag.plan_fragments(250, [133, 133, 133])
    assert sizes == [100, 100, 50]


def test_plan_single_carrier_exact_fit():
    assert frag.plan_fragments(100, [133]) == [100]


def test_plan_rejects_when_header_eats_capacity():
    # Raw capacity looks enough (120 >= 100) but header overhead makes it fail.
    with pytest.raises(CapacityError, match="fragment header"):
        frag.plan_fragments(100, [120])


def test_plan_error_reports_shortfall():
    with pytest.raises(CapacityError, match="Short by 50"):
        frag.plan_fragments(350, [133, 133, 133])


def test_plan_no_carriers():
    with pytest.raises(CapacityError):
        frag.plan_fragments(10, [])


# ── Split / join roundtrip ────────────────────────────────────────────────────

def test_roundtrip_across_carriers():
    caps = [1033, 1033, 1033]  # 1000 usable each
    frags = frag.split_payload(PAYLOAD, caps)
    used = [f for f in frags if f is not None]
    assert len(used) == 3
    assert all(len(f) <= c for f, c in zip(used, caps))
    assert frag.join_fragments(used) == PAYLOAD


def test_join_accepts_out_of_order_input():
    frags = [f for f in frag.split_payload(PAYLOAD, [1033, 1033, 1033]) if f]
    shuffled = [frags[2], frags[0], frags[1]]
    assert frag.join_fragments(shuffled) == PAYLOAD


def test_unused_carriers_get_none():
    frags = frag.split_payload(b"tiny", [500, 500, 500])
    assert frags[0] is not None
    assert frags[1] is None and frags[2] is None
    assert frag.join_fragments([frags[0]]) == b"tiny"


def test_same_split_shares_payload_id():
    frags = [f for f in frag.split_payload(PAYLOAD, [1033, 1033, 1033]) if f]
    ids = {frag.parse_fragment(f)["payload_id"] for f in frags}
    assert len(ids) == 1


# ── Decode-time errors ────────────────────────────────────────────────────────

def _frags(caps=(1033, 1033, 1033)):
    return [f for f in frag.split_payload(PAYLOAD, list(caps)) if f]


def test_missing_fragment_error():
    frags = _frags()
    with pytest.raises(FragmentError, match=r"Missing fragment\(s\): 1 \(have 2 of 3\)"):
        frag.join_fragments([frags[0], frags[2]])


def test_duplicate_fragment_error():
    frags = _frags()
    with pytest.raises(FragmentError, match="Duplicate fragment #0"):
        frag.join_fragments([frags[0], frags[0], frags[1], frags[2]])


def test_truncated_fragment_error():
    frags = _frags()
    with pytest.raises(FragmentError, match="truncated"):
        frag.join_fragments([frags[0][:-10], frags[1], frags[2]])


def test_truncated_header_error():
    with pytest.raises(FragmentError, match="header incomplete"):
        frag.parse_fragment(b"SFRF\x01\x00")


def test_corrupted_fragment_error():
    frags = _frags()
    bad = bytearray(frags[1])
    bad[-1] ^= 0xFF  # flip a data byte → CRC32 mismatch
    with pytest.raises(FragmentError, match="corrupted: CRC32 mismatch"):
        frag.join_fragments([frags[0], bytes(bad), frags[2]])


def test_mixed_payload_id_error():
    a = _frags()
    b = _frags()
    with pytest.raises(FragmentError, match="different payloads"):
        frag.join_fragments([a[0], b[1], a[2]])


def test_join_empty_error():
    with pytest.raises(FragmentError, match="No fragments"):
        frag.join_fragments([])


# ── CLI-level multi-carrier roundtrip ─────────────────────────────────────────

def _make_png(path: Path, size=(60, 60)):
    from PIL import Image
    Image.new("RGB", size, (120, 60, 30)).save(path, format="PNG")


CLI_PAYLOAD = bytes(range(256)) * 12  # 3072 bytes → 3 fragments at 60x60


def test_cli_multi_carrier_roundtrip(tmp_path, monkeypatch):
    from stegoforge import op_encode, op_decode

    monkeypatch.delenv("STEGOFORGE_KEY", raising=False)
    carriers = []
    for i in range(3):
        p = tmp_path / f"carrier{i}.png"
        _make_png(p)
        carriers.append(str(p))
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(CLI_PAYLOAD)

    result = op_encode(
        carriers, str(payload_path), "secret", None, "lsb", 1,
        None, None, False, None, False, False, None, None, None, True,
    )
    assert result["fragmented"] is True
    assert result["fragments"] == 3
    outputs = result["outputs"]
    assert all(Path(o).exists() for o in outputs)

    decoded = op_decode(outputs, "secret", None, "lsb", False, True)
    assert Path(decoded["output"]).read_bytes() == CLI_PAYLOAD


def test_cli_multi_carrier_out_of_order_decode(tmp_path, monkeypatch):
    from stegoforge import op_encode, op_decode

    monkeypatch.delenv("STEGOFORGE_KEY", raising=False)
    carriers = []
    for i in range(3):
        p = tmp_path / f"carrier{i}.png"
        _make_png(p)
        carriers.append(str(p))
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(CLI_PAYLOAD)

    result = op_encode(
        carriers, str(payload_path), None, None, "lsb", 1,
        None, None, False, None, False, False, None, None, None, True,
    )
    outputs = result["outputs"]
    decoded = op_decode([outputs[2], outputs[0], outputs[1]], None, None, "lsb", False, True)
    assert Path(decoded["output"]).read_bytes() == CLI_PAYLOAD


def test_cli_preflight_error_leaves_no_files(tmp_path, monkeypatch):
    from stegoforge import op_encode

    monkeypatch.delenv("STEGOFORGE_KEY", raising=False)
    carriers = []
    for i in range(2):
        p = tmp_path / f"carrier{i}.png"
        _make_png(p)
        carriers.append(str(p))
    payload_path = tmp_path / "big.bin"
    payload_path.write_bytes(b"x" * 100_000)  # way over capacity

    before = set(tmp_path.iterdir())
    with pytest.raises(CapacityError, match="Payload too large for carrier set"):
        op_encode(
            carriers, str(payload_path), None, None, "lsb", 1,
            None, None, False, None, False, False, None, None, None, True,
        )
    assert set(tmp_path.iterdir()) == before  # no partial output files


def test_cli_decode_missing_fragment_error(tmp_path, monkeypatch):
    from stegoforge import op_encode, op_decode

    monkeypatch.delenv("STEGOFORGE_KEY", raising=False)
    carriers = []
    for i in range(3):
        p = tmp_path / f"carrier{i}.png"
        _make_png(p)
        carriers.append(str(p))
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(CLI_PAYLOAD)

    result = op_encode(
        carriers, str(payload_path), None, None, "lsb", 1,
        None, None, False, None, False, False, None, None, None, True,
    )
    outputs = result["outputs"]
    with pytest.raises(FragmentError, match=r"Missing fragment\(s\): 1"):
        op_decode([outputs[0], outputs[2]], None, None, "lsb", False, True)
