"""JSON report generation for detected regions."""

import json
from pathlib import Path
from typing import Any


def write_report(
    output_path: Path,
    frames: list[dict[str, Any]],
) -> None:
    """Write JSON report with timestamps and regions per frame."""
    data = {"frames": frames}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
