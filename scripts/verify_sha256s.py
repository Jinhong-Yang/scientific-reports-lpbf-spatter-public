"""Verify a release SHA256SUMS ledger relative to the project root."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, ledger: Path) -> list[str]:
    failures = []
    for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        path = root / relative
        if not separator or len(digest) != 64 or not path.is_file() or sha256_file(path) != digest:
            failures.append(f"line {line_number}: {relative or line}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    ledger = args.ledger.resolve()
    root = ledger.parents[1] if ledger.parent.name == "release" else ledger.parent
    failures = verify(root, ledger)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"PASS {sum(1 for line in ledger.read_text(encoding='utf-8').splitlines() if line.strip())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
