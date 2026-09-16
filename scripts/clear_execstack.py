#!/usr/bin/env python3
"""Clear the executable-stack flag (PF_X in PT_GNU_STACK) from ELF shared libs.

The PyPI ``xtb`` wheel (xtb-python 22.1) bundles Fortran libraries built with an
executable stack. On modern kernels the dynamic loader refuses to honor that
("cannot enable executable stack as shared object requires"), so
``xtb.ase.calculator`` fails to import. ``execstack -c`` would fix it, but that
tool was dropped from recent Debian, so this pure-Python patcher clears the flag
directly. Pass one or more files or directories (directories are scanned for
``*.so*``).
"""
from __future__ import annotations

import glob
import os
import struct
import sys

PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def clear_execstack(path: str) -> bool:
    """Clear PF_X on the PT_GNU_STACK header of one ELF file. Returns True if changed."""
    with open(path, "r+b") as f:
        data = bytearray(f.read())
        if data[:4] != b"\x7fELF" or data[4] != 2:  # ELF64 only
            return False
        end = "<" if data[5] == 1 else ">"
        e_phoff = struct.unpack_from(end + "Q", data, 0x20)[0]
        e_phentsize = struct.unpack_from(end + "H", data, 0x36)[0]
        e_phnum = struct.unpack_from(end + "H", data, 0x38)[0]
        changed = False
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            if struct.unpack_from(end + "I", data, off)[0] == PT_GNU_STACK:
                flags = struct.unpack_from(end + "I", data, off + 4)[0]
                if flags & PF_X:
                    struct.pack_into(end + "I", data, off + 4, flags & ~PF_X)
                    changed = True
        if changed:
            f.seek(0)
            f.write(data)
        return changed


def iter_targets(args):
    for a in args:
        if os.path.isdir(a):
            yield from glob.glob(os.path.join(a, "**", "*.so*"), recursive=True)
        else:
            yield a


def main(argv):
    if not argv:
        print("usage: clear_execstack.py <file-or-dir> ...", file=sys.stderr)
        return 2
    n = 0
    for path in iter_targets(argv):
        try:
            if clear_execstack(path):
                print("cleared execstack:", path)
                n += 1
        except Exception as e:  # noqa: BLE001 - best-effort, keep going
            print(f"skip {path}: {e}", file=sys.stderr)
    print(f"patched {n} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
