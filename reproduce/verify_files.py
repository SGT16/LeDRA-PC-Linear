import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "file_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append({"path": item["path"], "reason": "missing or invalid path"})
            continue
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if value != item["sha256"] or path.stat().st_size != item["bytes"]:
            failures.append({"path": item["path"], "reason": "checksum or size mismatch"})
    print(json.dumps({"state": "passed" if not failures else "failed", "checked_files": len(manifest["files"]), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
