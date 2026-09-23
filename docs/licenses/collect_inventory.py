"""導入済みlock対象の配布物メタデータと権利文書をローカルで採取する。"""

from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re


def main():
    project = Path(__file__).resolve().parents[2]
    destination = project / "docs" / "licenses"
    lock_path = project / "requirements-lock.txt"
    items = []
    for spec in lock_path.read_text(encoding="utf-8-sig").splitlines():
        if not spec or spec.startswith("#"):
            continue
        name, expected_version = spec.split("==", 1)
        distribution = metadata.distribution(name)
        if distribution.version != expected_version:
            raise ValueError(f"lockと導入版が不一致: {name}")
        key = re.sub(r"[-_.]+", "-", name).lower()
        item = {
            "name": distribution.metadata["Name"], "version": distribution.version,
            "license_expression": distribution.metadata.get("License-Expression"),
            "license_metadata": distribution.metadata.get("License"),
            "license_classifiers": [value for value in distribution.metadata.get_all("Classifier", []) if value.startswith("License ::")],
            "license_file_headers": distribution.metadata.get_all("License-File", []),
            "project_urls": distribution.metadata.get_all("Project-URL", []),
            "license_documents": [], "installed_binary_files": [],
        }
        for member in sorted(distribution.files or [], key=str):
            path = Path(str(member))
            if path.suffix.lower() in (".dll", ".pyd", ".so", ".dylib"):
                item["installed_binary_files"].append(path.as_posix())
            if not path.name.lower().startswith(("license", "licence", "copying", "copyright", "notice", "authors")):
                continue
            if path.suffix.lower() in (".py", ".pyc", ".pyd", ".dll", ".so") or ".." in path.parts or path.is_absolute():
                continue
            source = distribution.locate_file(member)
            if not source.is_file():
                continue
            content = source.read_bytes()
            target = destination / key / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f"既存権利文書と相違: {target.name}")
            if not target.exists():
                target.write_bytes(content)
            item["license_documents"].append({
                "installed_relative_path": path.as_posix(),
                "saved_relative_path": target.relative_to(project).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
            })
        items.append(item)
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "installed distributions listed in requirements-lock.txt; license texts only are copied, no library code or binaries",
        "requirements_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "distributions": items,
    }
    (destination / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"distributions={len(items)}, documents={sum(len(item['license_documents']) for item in items)}")
    print("license_document_missing=" + ", ".join(item["name"] for item in items if not item["license_documents"]))


if __name__ == "__main__":
    main()
