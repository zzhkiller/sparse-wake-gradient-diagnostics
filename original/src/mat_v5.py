"""Minimal read-only MAT-v5 numeric reader for these author's simple arrays.

Supports compressed, real dense numeric matrices only. It intentionally rejects
structs, cells, sparse/complex arrays and v7.3 files; it never executes file content.
The binary array shape/order is checked again by audit_data.load_data.
"""
from __future__ import annotations

from pathlib import Path
import struct
import zlib
import numpy as np

TYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4",
         7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}


def element(buf, pos):
    tag, = struct.unpack_from("<I", buf, pos)
    small_size = tag >> 16
    if small_size:
        if small_size > 4:
            raise ValueError("Invalid small data element")
        return tag & 0xffff, buf[pos + 4:pos + 4 + small_size], pos + 8
    kind, size = struct.unpack_from("<II", buf, pos)
    end = pos + 8 + size
    if end > len(buf):
        raise ValueError("Truncated MAT element")
    return kind, buf[pos + 8:end], end + ((-size) % 8)


def matrix(buf):
    kind, payload, _ = element(buf, 0)
    if kind != 14:
        raise ValueError(f"Expected miMATRIX, received {kind}")
    kind, flags, pos = element(payload, 0)
    class_flags, = struct.unpack_from("<I", flags)
    if kind != 6 or (class_flags & 0xff) not in range(6, 16) or class_flags & 0x800:
        raise ValueError("Only real dense numeric matrices are supported")
    kind, dims, pos = element(payload, pos)
    if kind != 5:
        raise ValueError("Expected int32 dimensions")
    shape = tuple(np.frombuffer(dims, dtype="<i4").tolist())
    _, name, pos = element(payload, pos)
    kind, values, _ = element(payload, pos)
    if kind not in TYPES:
        raise ValueError(f"Unsupported numeric type {kind}")
    array = np.frombuffer(values, dtype=TYPES[kind]).copy().reshape(shape, order="F")
    return name.decode("ascii"), array


def loadmat(path):
    arrays = {}
    with Path(path).open("rb") as stream:
        header = stream.read(128)
        if len(header) != 128 or header[-2:] != b"IM" or not header.startswith(b"MATLAB 5.0"):
            raise ValueError("Expected little-endian MATLAB v5 file")
        while tag := stream.read(8):
            if len(tag) != 8:
                raise ValueError("Truncated outer tag")
            kind, size = struct.unpack("<II", tag)
            body = stream.read(size)
            if len(body) != size:
                raise ValueError("Truncated outer element")
            if kind == 15:
                # Compressed top-level elements in these v5 files are unpadded.
                unpacked = zlib.decompress(body)
            elif kind == 14:
                unpacked = tag + body
                stream.read((-size) % 8)
            else:
                raise ValueError(f"Unsupported outer type {kind}")
            name, arr = matrix(unpacked)
            if name in arrays:
                raise ValueError(f"Duplicate variable {name}")
            arrays[name] = arr
    return arrays
