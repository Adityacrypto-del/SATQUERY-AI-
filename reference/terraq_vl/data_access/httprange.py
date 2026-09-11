"""Seekable read-only file over HTTP Range requests, plus a minimal LMDB reader on top."""
from __future__ import annotations

import io
import struct
import urllib.request

BLOCK = 1 << 16
# Without a timeout a stalled socket blocks a worker forever (seen: fetch hung at 1072/1082).
# The fetcher retries on timeout, so this only bounds how long one stall can last.
TIMEOUT_S = 60


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = BLOCK):
        self.url = url
        self.block = block
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        self.bytes_fetched = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            self.url = r.geturl()  # follow redirect once
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            data = r.read()
        self.bytes_fetched += len(data)
        return data

    def pread(self, off: int, n: int) -> bytes:
        n = max(0, min(n, self.size - off))
        if n > 4 * self.block:  # large contiguous read: fetch directly
            return self._fetch(off, off + n)
        out = bytearray()
        b0, b1 = off // self.block, (off + n - 1) // self.block
        for b in range(b0, b1 + 1):
            if b not in self.cache:
                s = b * self.block
                self.cache[b] = self._fetch(s, min(s + self.block, self.size))
            out += self.cache[b]
        s = off - b0 * self.block
        return bytes(out[s:s + n])

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        d = self.pread(self.pos, n)
        self.pos += len(d)
        return d

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


# ---- LMDB (64-bit, default memcmp keys) -------------------------------------
P_BRANCH, P_LEAF, P_OVERFLOW, P_META, P_LEAF2 = 0x01, 0x02, 0x04, 0x08, 0x20
F_BIGDATA = 0x01
HDR = 16


class RemoteLMDB:
    def __init__(self, f: HTTPRangeFile, psize: int = 4096):
        self.f = f
        metas = []
        for p in (0, 1):
            pg = f.pread(p * psize, psize)
            magic, ver = struct.unpack_from("<II", pg, HDR)
            assert magic == 0xBEEFC0DE, hex(magic)
            off = HDR + 8 + 8 + 8
            free_pad = struct.unpack_from("<I", pg, off)[0]
            main = struct.unpack_from("<IHHQQQQQ", pg, off + 48)
            last_pg, txnid = struct.unpack_from("<QQ", pg, off + 96)
            metas.append((txnid, free_pad, main))
        txnid, free_pad, main = max(metas)
        self.psize = free_pad or psize
        self.depth, self.entries, self.root = main[2], main[6], main[7]

    def page(self, pgno: int) -> bytes:
        return self.f.pread(pgno * self.psize, self.psize)

    def _nodes(self, pg: bytes):
        lower = struct.unpack_from("<H", pg, HDR - 4)[0]
        n = (lower - HDR) // 2
        for i in range(n):
            o = struct.unpack_from("<H", pg, HDR + 2 * i)[0]
            lo, hi, fl, ks = struct.unpack_from("<HHHH", pg, o)
            key = pg[o + 8:o + 8 + ks]
            yield lo, hi, fl, key, o + 8 + ks

    def get(self, key: bytes) -> bytes | None:
        pgno = self.root
        while True:
            pg = self.page(pgno)
            flags = struct.unpack_from("<H", pg, 10)[0]
            nodes = list(self._nodes(pg))
            if flags & P_BRANCH:
                child = None
                for i, (lo, hi, fl, k, _) in enumerate(nodes):
                    if i == 0 or k <= key:
                        child = lo | (hi << 16) | (fl << 32)
                    else:
                        break
                pgno = child
                continue
            assert flags & P_LEAF and not flags & P_LEAF2, hex(flags)
            for lo, hi, fl, k, doff in nodes:
                if k == key:
                    dsize = lo | (hi << 16)
                    if fl & F_BIGDATA:
                        ov = struct.unpack_from("<Q", pg, doff)[0]
                        return self.f.pread(ov * self.psize + HDR, dsize)
                    return pg[doff:doff + dsize]
            return None
