"""tests/test_framing.py — Tests for chunked payload framing (core/framing.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core import framing
from core.framing import (
    CapacityError,
    DuplicateChunkError,
    FrameChecksumError,
    FrameTruncatedError,
    MissingChunkError,
    SessionMismatchError,
    FRAME_HEADER_SIZE,
)


# ── Frame pack/unpack ─────────────────────────────────────────────────────────

def test_frame_roundtrip():
    blob = framing.pack_frame(b"12345678", 2, 5, b"hello chunk")
    f = framing.unpack_frame(blob)
    assert f.session_id == b"12345678"
    assert f.index == 2
    assert f.total == 5
    assert f.data == b"hello chunk"


def test_frame_header_size_matches_struct():
    assert FRAME_HEADER_SIZE == 25
    blob = framing.pack_frame(b"12345678", 0, 1, b"")
    assert len(blob) == FRAME_HEADER_SIZE


def test_unpack_truncated_header():
    blob = framing.pack_frame(b"12345678", 0, 1, b"data")
    with pytest.raises(FrameTruncatedError, match="truncated"):
        framing.unpack_frame(blob[:10])


def test_unpack_truncated_data():
    blob = framing.pack_frame(b"12345678", 0, 1, b"x" * 100)
    with pytest.raises(FrameTruncatedError, match="declares 100"):
        framing.unpack_frame(blob[:-10])


def test_unpack_checksum_mismatch():
    blob = bytearray(framing.pack_frame(b"12345678", 0, 1, b"payload"))
    blob[-1] ^= 0xFF  # flip a bit in the data
    with pytest.raises(FrameChecksumError, match="checksum mismatch"):
        framing.unpack_frame(bytes(blob))


def test_unpack_bad_magic():
    blob = b"XXXX" + framing.pack_frame(b"12345678", 0, 1, b"data")[4:]
    with pytest.raises(ValueError, match="bad magic"):
        framing.unpack_frame(blob)


def test_is_frame():
    assert framing.is_frame(framing.pack_frame(b"12345678", 0, 1, b""))
    assert not framing.is_frame(b"random data here........")
    assert not framing.is_frame(b"")


# ── Capacity planning ─────────────────────────────────────────────────────────

def test_usable_capacity_subtracts_overhead():
    assert framing.usable_capacity(1000) == 1000 - FRAME_HEADER_SIZE
    assert framing.usable_capacity(10) == 0


def test_plan_chunks_fits_exactly():
    assert framing.plan_chunks(100, [50, 50]) == [50, 50]


def test_plan_chunks_partial_last_carrier():
    assert framing.plan_chunks(120, [100, 100]) == [100, 20]


def test_plan_chunks_skips_unneeded_carriers():
    assert framing.plan_chunks(10, [100, 100, 100]) == [10]


def test_plan_chunks_overflow_raises_before_write():
    with pytest.raises(CapacityError, match="Short by 50"):
        framing.plan_chunks(150, [100])


def test_plan_chunks_no_carriers():
    with pytest.raises(CapacityError):
        framing.plan_chunks(10, [])


def test_split_payload_reassembles():
    payload = bytes(range(256)) * 3
    chunks = framing.split_payload(payload, [200, 200, 368])
    assert b"".join(chunks) == payload
    assert [len(c) for c in chunks] == [200, 200, 368]


# ── Reassembly ────────────────────────────────────────────────────────────────

def _frames(payload: bytes, caps: list[int], session: bytes = b"session1"):
    return [framing.unpack_frame(b) for b in framing.build_frames(payload, caps, session)]


def test_reassemble_in_order():
    payload = b"A" * 100 + b"B" * 100 + b"C" * 50
    frames = _frames(payload, [100, 100, 100])
    assert framing.reassemble(frames) == payload


def test_reassemble_out_of_order_input():
    payload = bytes(range(256))
    frames = _frames(payload, [100, 100, 56])
    shuffled = [frames[2], frames[0], frames[1]]
    assert framing.reassemble(shuffled) == payload


def test_reassemble_missing_chunk():
    frames = _frames(b"x" * 250, [100, 100, 100])
    with pytest.raises(MissingChunkError, match="Missing 1 of 3 chunk"):
        framing.reassemble([frames[0], frames[2]])


def test_reassemble_missing_all():
    with pytest.raises(MissingChunkError, match="No chunks"):
        framing.reassemble([])


def test_reassemble_duplicate_chunk():
    frames = _frames(b"x" * 200, [100, 100])
    with pytest.raises(DuplicateChunkError, match="more than once"):
        framing.reassemble([frames[0], frames[0]])


def test_reassemble_session_mismatch():
    a = _frames(b"x" * 100, [100], session=b"sessionA")
    b = _frames(b"y" * 100, [100], session=b"sessionB")
    with pytest.raises(SessionMismatchError, match="different encode sessions"):
        framing.reassemble(a + b)


def test_reassemble_total_mismatch():
    session = b"session1"
    f1 = framing.unpack_frame(framing.pack_frame(session, 0, 2, b"a" * 10))
    f2 = framing.unpack_frame(framing.pack_frame(session, 1, 3, b"b" * 10))
    with pytest.raises(SessionMismatchError, match="disagree on total"):
        framing.reassemble([f1, f2])


# ── Errors subclass ValueError (CLI catches them) ─────────────────────────────

def test_errors_are_value_errors():
    for exc in (CapacityError, FrameTruncatedError, FrameChecksumError,
                MissingChunkError, DuplicateChunkError, SessionMismatchError):
        assert issubclass(exc, ValueError)
