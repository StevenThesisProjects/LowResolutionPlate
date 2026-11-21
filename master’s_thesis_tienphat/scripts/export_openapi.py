#!/usr/bin/env python3
"""Write the API's OpenAPI spec to artifacts/openapi.json (roadmap week 7: "OpenAPI spec").

    python3 scripts/export_openapi.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mfplate.api import app  # noqa: E402
from mfplate.settings import ARTIFACTS_DIR  # noqa: E402

spec = app.openapi()
out = ARTIFACTS_DIR / "openapi.json"
out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print(f"📝 {out}: {len(spec['paths'])} endpoint, phiên bản {spec['info']['version']}")
