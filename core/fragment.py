"""
core/fragment.py — Multi-carrier payload fragmentation.

Splits a large (already encrypted) payload across several carriers and
reassembles it in order on decode. Each fragment is a self-describing
container embedded as the encoder payload of one carrier.

Wire format (33-byte header + fragment data):
  [4 bytes  magic: b'SFRF']
  [1 byte   version: 1]
  [2 bytes  frag_index: uint16 BE]
  [2 bytes  frag_count: uint16 BE]
  [8 bytes  payload_id: uint64 BE — random per split session]
  [4 bytes  total_len: uint32 BE — reassembled payload length]
  [4 bytes  payload_crc32: CRC32 of the full reassembled payload]
  [4 bytes  frag_len: uint32 BE — this fragment's data length]
  [4 bytes  frag_crc32: CRC32 of this fragment's data]
  [N bytes  fragment data]

Capacity contract:
  usable bytes per carrier = encoder.capacity(...) - HEADER_SIZE
  plan_fragments() accounts for this overhead up front, so an
  over-capacity job fails before any carrier is touched.
"""
import os
import struct
import zlib

MAGIC = b"SFRF"
VERSION = 1
HEADER_FMT = ">4sBHHQIIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 33 bytes
MAX_FRAGMENTS = 0xFFFF


class FragmentError(ValueError):
    """Raised when fragments cannot be parsed or reassembled."""


class CapacityError(ValueError):
    """Raised when the payload does not fit the planned carrier set."""


def usable_capacity(raw_capacity: int) -> int:
    """Payload bytes usable in one carrier after the fragment header."""
    return max(0, raw_capacity - HEADER_SIZE)


def plan_fragments(payload_len: int, capacities: list[int]) -> list[int]:
    """
    Distribute payload_len bytes across carriers with the given raw capacities.

    Returns a list of fragment data sizes aligned with `capacities`
    (0 = carrier unused). Raises CapacityError before anything is embedded
    if the total usable capacity is insufficient.
    """
    if payload_len < 0:
        raise CapacityError(f"Invalid payload length: {payload_len}")
    if not capacities:
        raise CapacityError("No carriers provided")
    if len(capacities) > MAX_FRAGMENTS:
        raise CapacityError(
            f"Too many carriers: {len(capacities)} (max {MAX_FRAGMENTS})"
        )

    usable = [usable_capacity(c) for c in capacities]
    total_usable = sum(usable)
    if payload_len > total_usable:
        raise CapacityError(
            f"Payload too large for carrier set: {payload_len:,} bytes needed, "
            f"{total_usable:,} bytes usable across {len(capacities)} carrier(s) "
            f"(raw capacity {sum(capacities):,} bytes minus {HEADER_SIZE} bytes "
            f"fragment header + CRC32 per carrier). "
            f"Short by {payload_len - total_usable:,} bytes — add more carriers "
            f"or use larger ones."
        )

    sizes = []
    remaining = payload_len
    for u in usable:
        take = min(u, remaining)
        sizes.append(take)
        remaining -= take
    if payload_len == 0:
        # Still need one (empty) fragment so decode can find the container.
        if not any(c >= HEADER_SIZE for c in capacities):
            raise CapacityError(
                f"No carrier can hold a {HEADER_SIZE}-byte fragment header"
            )
    return sizes


def split_payload(
    payload: bytes,
    capacities: list[int],
    payload_id: int | None = None,
) -> list[bytes | None]:
    """
    Split payload into fragment containers sized for the given raw capacities.

    Returns a list aligned with `capacities`: fragment bytes for used
    carriers, None for unused ones. Raises CapacityError up front if the
    payload does not fit.
    """
    sizes = plan_fragments(len(payload), capacities)
    if payload_id is None:
        payload_id = int.from_bytes(os.urandom(8), "big")
    total_crc = zlib.crc32(payload) & 0xFFFFFFFF
    frag_count = sum(1 for s in sizes if s > 0) or 1

    fragments: list[bytes | None] = []
    offset = 0
    frag_index = 0
    for size in sizes:
        if size == 0 and not (len(payload) == 0 and frag_index == 0):
            fragments.append(None)
            continue
        data = payload[offset:offset + size]
        offset += size
        header = struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            frag_index,
            frag_count,
            payload_id,
            len(payload),
            total_crc,
            len(data),
            zlib.crc32(data) & 0xFFFFFFFF,
        )
        fragments.append(header + data)
        frag_index += 1
    return fragments


def is_fragment(data: bytes) -> bool:
    """Return True if data looks like a fragment container."""
    return len(data) >= HEADER_SIZE and data[:4] == MAGIC


def parse_fragment(data: bytes) -> dict:
    """
    Parse and validate one fragment container.

    Raises FragmentError if the fragment is truncated or corrupted.
    """
    if len(data) < HEADER_SIZE:
        raise FragmentError(
            f"Fragment truncated: header incomplete "
            f"(need {HEADER_SIZE} bytes, got {len(data)})"
        )
    (magic, version, frag_index, frag_count, payload_id,
     total_len, total_crc, frag_len, frag_crc) = struct.unpack(
        HEADER_FMT, data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise FragmentError("Not a fragment container (bad magic)")
    if version != VERSION:
        raise FragmentError(f"Unsupported fragment version: {version}")
    if frag_count == 0:
        raise FragmentError("Invalid fragment header: frag_count is 0")
    if frag_index >= frag_count:
        raise FragmentError(
            f"Invalid fragment header: index {frag_index} out of range "
            f"(frag_count {frag_count})"
        )
    body = data[HEADER_SIZE:]
    if len(body) != frag_len:
        raise FragmentError(
            f"Fragment #{frag_index} truncated: declares {frag_len:,} data "
            f"bytes but contains {len(body):,}"
        )
    actual_crc = zlib.crc32(body) & 0xFFFFFFFF
    if actual_crc != frag_crc:
        raise FragmentError(
            f"Fragment #{frag_index} corrupted: CRC32 mismatch "
            f"(expected {frag_crc:#010x}, got {actual_crc:#010x})"
        )
    return {
        "index": frag_index,
        "count": frag_count,
        "payload_id": payload_id,
        "total_len": total_len,
        "total_crc": total_crc,
        "data": body,
    }


def join_fragments(fragments: list[bytes]) -> bytes:
    """
    Reassemble the original payload from fragment containers, in order.

    Fragments may be supplied out of order; they are sorted by index.
    Raises FragmentError with a specific message for missing, duplicate,
    mixed-session, truncated, or corrupted fragments.
    """
    if not fragments:
        raise FragmentError("No fragments provided")

    parsed = [parse_fragment(f) for f in fragments]

    ids = {p["payload_id"] for p in parsed}
    if len(ids) > 1:
        raise FragmentError(
            f"Fragments belong to {len(ids)} different payloads "
            f"(mixed payload IDs) — cannot reassemble"
        )

    counts = {p["count"] for p in parsed}
    totals = {(p["total_len"], p["total_crc"]) for p in parsed}
    if len(counts) > 1 or len(totals) > 1:
        raise FragmentError(
            "Inconsistent fragment headers: fragments disagree on "
            "frag_count or total payload size/checksum"
        )
    count = counts.pop()
    total_len, total_crc = totals.pop()

    by_index: dict[int, dict] = {}
    for p in parsed:
        if p["index"] in by_index:
            raise FragmentError(f"Duplicate fragment #{p['index']}")
        by_index[p["index"]] = p

    missing = [i for i in range(count) if i not in by_index]
    if missing:
        raise FragmentError(
            f"Missing fragment(s): {', '.join(map(str, missing))} "
            f"(have {count - len(missing)} of {count})"
        )

    payload = b"".join(by_index[i]["data"] for i in range(count))
    if len(payload) != total_len:
        raise FragmentError(
            f"Reassembled payload truncated: expected {total_len:,} bytes, "
            f"got {len(payload):,}"
        )
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != total_crc:
        raise FragmentError(
            f"Reassembled payload corrupted: CRC32 mismatch "
            f"(expected {total_crc:#010x}, got {actual_crc:#010x})"
        )
    return payload
