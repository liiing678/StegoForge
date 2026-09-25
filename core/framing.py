"""
core/framing.py — Chunked payload framing for multi-carrier embedding.

A payload that does not fit in a single carrier is split into chunks.
Each chunk is wrapped in a frame before being handed to an encoder, so
the capacity estimate must subtract the frame overhead:

Wire format of one frame (big-endian):
  [4 bytes  magic: b'SFCH']
  [1 byte   version: 0x01]
  [8 bytes  session_id]      random per encode session, ties chunks together
  [2 bytes  chunk_index]     0-based
  [2 bytes  total_chunks]
  [4 bytes  data_length]     length of the chunk data in this frame
  [4 bytes  crc32]           CRC32 of everything above + chunk data
  [N bytes  chunk data]

FRAME_HEADER_SIZE = 25 bytes. The encoder's own length header is already
excluded by BaseEncoder.capacity(), so the usable payload per carrier is:

  usable = encoder.capacity(carrier) - FRAME_HEADER_SIZE

All errors raised here subclass ValueError so the CLI error path catches them.
"""
import os
import struct
import zlib
from dataclasses import dataclass

MAGIC = b"SFCH"
VERSION = 1
HEADER_FMT = ">4sB8sHHI I".replace(" ", "")  # magic, version, session, index, total, data_len, crc32
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 25 bytes
FRAME_HEADER_SIZE = HEADER_SIZE

MAX_CHUNKS = 0xFFFF  # uint16 field width


# ── Errors ────────────────────────────────────────────────────────────────────

class CapacityError(ValueError):
    """Payload does not fit in the planned carrier(s). Raised before any write."""


class FrameError(ValueError):
    """Base class for frame parsing / reassembly errors."""


class FrameTruncatedError(FrameError):
    """Frame is shorter than its header or declared data length."""


class FrameChecksumError(FrameError):
    """Frame CRC32 mismatch — data corrupted or truncated."""


class MissingChunkError(FrameError):
    """One or more chunk indices are absent during reassembly."""


class DuplicateChunkError(FrameError):
    """The same chunk index appeared more than once."""


class SessionMismatchError(FrameError):
    """Chunks belong to different encode sessions or disagree on total count."""


# ── Frame packing ─────────────────────────────────────────────────────────────

@dataclass
class Frame:
    session_id: bytes
    index: int
    total: int
    data: bytes


def pack_frame(session_id: bytes, index: int, total: int, data: bytes) -> bytes:
    """Wrap one chunk in a frame. `session_id` must be exactly 8 bytes."""
    if len(session_id) != 8:
        raise ValueError(f"session_id must be 8 bytes, got {len(session_id)}")
    if not (0 <= index < total <= MAX_CHUNKS):
        raise ValueError(f"invalid chunk coordinates: index={index} total={total}")
    crc = zlib.crc32(session_id + struct.pack(">HHI", index, total, len(data)) + data)
    header = struct.pack(HEADER_FMT, MAGIC, VERSION, session_id, index, total, len(data), crc)
    return header + data


def is_frame(data: bytes) -> bool:
    """Return True if data looks like a framed chunk."""
    return len(data) >= HEADER_SIZE and data[:4] == MAGIC


def unpack_frame(blob: bytes) -> Frame:
    """
    Parse and validate a frame.

    Raises:
        FrameTruncatedError: blob shorter than header or declared data length.
        FrameChecksumError:  CRC32 mismatch (corrupted or truncated data).
        FrameError:          bad magic or unsupported version.
    """
    if len(blob) < HEADER_SIZE:
        raise FrameTruncatedError(
            f"Frame truncated: {len(blob)} bytes, need at least {HEADER_SIZE} for header"
        )
    magic, version, session_id, index, total, data_len, crc = struct.unpack(
        HEADER_FMT, blob[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise FrameError("Not a StegoForge chunk frame (bad magic)")
    if version != VERSION:
        raise FrameError(f"Unsupported frame version: {version}")
    if total == 0 or total > MAX_CHUNKS or index >= total:
        raise FrameError(f"Invalid chunk coordinates in frame: index={index} total={total}")
    if len(blob) < HEADER_SIZE + data_len:
        raise FrameTruncatedError(
            f"Frame truncated: declares {data_len} data bytes, "
            f"only {len(blob) - HEADER_SIZE} present"
        )
    data = blob[HEADER_SIZE:HEADER_SIZE + data_len]
    expect = zlib.crc32(session_id + struct.pack(">HHI", index, total, data_len) + data)
    if crc != expect:
        raise FrameChecksumError(
            f"Frame checksum mismatch on chunk {index + 1}/{total} "
            f"(corrupted or truncated carrier)"
        )
    return Frame(session_id=session_id, index=index, total=total, data=data)


# ── Capacity planning ─────────────────────────────────────────────────────────

def usable_capacity(encoder_capacity: int) -> int:
    """
    Convert a raw encoder capacity into usable payload bytes per carrier,
    subtracting the frame header + checksum overhead.
    """
    return max(0, encoder_capacity - FRAME_HEADER_SIZE)


def plan_chunks(payload_size: int, capacities: list[int]) -> list[int]:
    """
    Given the total payload size and the *usable* capacity of each carrier
    (in order), return the chunk size assigned to each carrier.

    Raises CapacityError before anything is written if the payload does not
    fit, or if too many carriers would be needed.
    """
    if payload_size < 0:
        raise ValueError("payload_size must be >= 0")
    if not capacities:
        raise CapacityError("No carriers provided")
    if len(capacities) > MAX_CHUNKS:
        raise CapacityError(
            f"Too many carriers: {len(capacities)} (max {MAX_CHUNKS} chunks)"
        )

    plan: list[int] = []
    remaining = payload_size
    for cap in capacities:
        if remaining <= 0:
            break
        take = min(cap, remaining)
        plan.append(take)
        remaining -= take

    if remaining > 0:
        total = sum(capacities)
        raise CapacityError(
            f"Payload too large for provided carriers: {payload_size:,} bytes needed, "
            f"{total:,} bytes usable across {len(capacities)} carrier(s) "
            f"(after {FRAME_HEADER_SIZE}-byte frame header + CRC32 overhead per carrier). "
            f"Short by {remaining:,} bytes — add more/larger carriers, raise --depth, "
            f"or compress the payload."
        )
    return plan


def split_payload(payload: bytes, capacities: list[int]) -> list[bytes]:
    """Split payload into one chunk per carrier according to plan_chunks()."""
    plan = plan_chunks(len(payload), capacities)
    chunks = []
    offset = 0
    for size in plan:
        chunks.append(payload[offset:offset + size])
        offset += size
    return chunks


def new_session_id() -> bytes:
    """Generate a random 8-byte session id for one multi-carrier encode."""
    return os.urandom(8)


def build_frames(payload: bytes, capacities: list[int], session_id: bytes | None = None) -> list[bytes]:
    """Split payload and return packed frames, one per carrier used."""
    session_id = session_id or new_session_id()
    chunks = split_payload(payload, capacities)
    total = len(chunks)
    return [pack_frame(session_id, i, total, chunk) for i, chunk in enumerate(chunks)]


# ── Reassembly ────────────────────────────────────────────────────────────────

def reassemble(frames: list[Frame]) -> bytes:
    """
    Reassemble the original payload from parsed frames, in chunk order.

    Input order does not matter — frames are sorted by their embedded index.

    Raises:
        SessionMismatchError: frames from different sessions, or disagreeing
                              total chunk counts.
        DuplicateChunkError:  same chunk index supplied twice.
        MissingChunkError:    gaps in the 0..total-1 index sequence.
    """
    if not frames:
        raise MissingChunkError("No chunks provided — nothing to reassemble")

    sessions = {f.session_id for f in frames}
    if len(sessions) > 1:
        raise SessionMismatchError(
            f"Chunks come from {len(sessions)} different encode sessions; "
            f"only pass carriers produced by the same encode-multi run"
        )
    totals = {f.total for f in frames}
    if len(totals) > 1:
        raise SessionMismatchError(
            f"Chunks disagree on total chunk count: {sorted(totals)} — "
            f"carriers are from different runs or one is corrupted"
        )
    total = totals.pop()

    by_index: dict[int, Frame] = {}
    for f in frames:
        if f.index in by_index:
            raise DuplicateChunkError(
                f"Chunk {f.index + 1}/{total} supplied more than once"
            )
        by_index[f.index] = f

    missing = [i for i in range(total) if i not in by_index]
    if missing:
        shown = ", ".join(str(i + 1) for i in missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise MissingChunkError(
            f"Missing {len(missing)} of {total} chunk(s): "
            f"chunk number(s) {shown}{more} not provided"
        )

    return b"".join(by_index[i].data for i in range(total))
