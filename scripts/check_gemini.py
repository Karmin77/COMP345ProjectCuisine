"""Send one bounded embedding request; never write the API key to disk."""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from comp345.embeddings import MODEL_NAME, OUTPUT_DIMENSIONALITY, TASK_TYPE


def main() -> int:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or getpass.getpass("Gemini API key (hidden): ")
    started = time.monotonic()
    try:
        with genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=20000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) as client:
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents="garlic\nsalt",
                config=types.EmbedContentConfig(
                    task_type=TASK_TYPE,
                    output_dimensionality=OUTPUT_DIMENSIONALITY,
                ),
            )
        vectors = response.embeddings or []
        vector = np.asarray(vectors[0].values if vectors else [], dtype=np.float32)
        valid = vector.shape == (OUTPUT_DIMENSIONALITY,) and bool(np.isfinite(vector).all())
        print(json.dumps({
            "ok": valid,
            "model_requested": MODEL_NAME,
            "task_type": TASK_TYPE,
            "dimensions": int(vector.size),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }), flush=True)
        return 0 if valid else 1
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error_type": type(exc).__name__,
            "status_code": getattr(exc, "code", None),
            "message": str(exc).replace(key, "[REDACTED]")[:2500],
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
