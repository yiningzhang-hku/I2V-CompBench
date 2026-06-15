"""
Phase 2 SiliconFlow client.

Subclasses Phase 1's SiliconFlowClient (now in i2vcompbench.utils.api_client_phase1) and adds:
- call_t2i / async_call_t2i  -> SiliconFlow `/v1/images/generations` (OpenAI-compatible)
- call_vqa_structured        -> wraps call_vlm and parses strict JSON QC checks

If the Phase 1 client cannot be imported for some reason, falls back to a
self-contained implementation that mirrors the same constructor + call_vlm/call_llm
interface.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image

# Try to reuse Phase 1 client (now part of the same package)
try:
    from .api_client_phase1 import SiliconFlowClient as _Phase1Client  # type: ignore
    _HAS_PHASE1 = True
except Exception as e:  # pragma: no cover
    logger.warning(f"Phase 1 SiliconFlowClient not importable ({e}); using local fallback")
    _HAS_PHASE1 = False
    _Phase1Client = object  # type: ignore


def _load_b64(path: str) -> str:
    img = Image.open(path)
    if img.mode not in ("RGB",):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > 2048:
        scale = 2048 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# Local fallback (only used if Phase 1 is unreachable)
# ============================================================

class _LocalClient:
    def __init__(self, config: dict):
        from openai import AsyncOpenAI, OpenAI

        api_cfg = config["api"]
        api_key = os.environ.get(api_cfg["api_key_env"], "")
        self.base_url = api_cfg["base_url"]
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        self.vlm_model = api_cfg["vlm"]["model"]
        self.vlm_max_tokens = api_cfg["vlm"].get("max_tokens", 2048)
        self.vlm_temperature = api_cfg["vlm"].get("temperature", 0.0)
        self.llm_model = api_cfg["llm"]["model"]
        self.llm_max_tokens = api_cfg["llm"].get("max_tokens", 1500)
        self.llm_temperature = api_cfg["llm"].get("temperature", 0.0)
        self.batch_size = api_cfg.get("batch_size", 5)
        self.retry_count = api_cfg.get("retry_count", 3)
        self.retry_delay = api_cfg.get("retry_delay", 2)
        self.timeout = api_cfg.get("timeout", 120)
        self.rate_limit_delay = api_cfg.get("rate_limit_delay", 0.5)
        self._api_key = api_key
        self._api_cfg = api_cfg

    def call_vlm(self, image_path: str, prompt_text: str) -> str:
        try:
            b64 = _load_b64(image_path)
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return ""
        for attempt in range(self.retry_count):
            try:
                r = self.client.chat.completions.create(
                    model=self.vlm_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }],
                    max_tokens=self.vlm_max_tokens,
                    temperature=self.vlm_temperature,
                    timeout=self.timeout,
                )
                time.sleep(self.rate_limit_delay)
                return r.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"VLM attempt {attempt+1} failed: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))
        return ""

    def call_llm(self, prompt_text: str) -> str:
        for attempt in range(self.retry_count):
            try:
                r = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=self.llm_max_tokens,
                    temperature=self.llm_temperature,
                    timeout=self.timeout,
                )
                time.sleep(self.rate_limit_delay)
                return r.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1} failed: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))
        return ""


_BaseClient = _Phase1Client if _HAS_PHASE1 else _LocalClient


# ============================================================
# Phase 2 client
# ============================================================

class Phase2SiliconFlowClient(_BaseClient):  # type: ignore[misc, valid-type]
    """SiliconFlow client extended with T2I + structured VQA helpers."""

    def __init__(self, config: dict):
        super().__init__(config)
        api_cfg = config["api"]
        t2i_cfg = api_cfg.get("t2i", {})
        self.t2i_model: str = t2i_cfg.get("model", "Kwai-Kolors/Kolors")
        self.t2i_size: str = t2i_cfg.get("image_size", "1024x1024")
        self.t2i_n: int = t2i_cfg.get("n", 1)
        self.t2i_endpoint: str = t2i_cfg.get(
            "endpoint", f"{self.base_url.rstrip('/')}/images/generations"
        )
        # api_key resolution (Phase 1 client keeps it on self.client; reach through env)
        self._t2i_api_key = os.environ.get(api_cfg["api_key_env"], "")
        logger.info(f"Phase2SiliconFlowClient: T2I model={self.t2i_model}, size={self.t2i_size}")

    # ---------------- T2I ----------------
    def call_t2i(
        self,
        prompt: str,
        negative: str = "",
        n: Optional[int] = None,
        size: Optional[str] = None,
    ) -> List[bytes]:
        """
        Synchronous T2I call. Returns a list of PNG/JPEG bytes (length == n).

        SiliconFlow's image generation endpoint is OpenAI-compatible. The exact response
        shape may vary across models: it commonly returns either base64 (`b64_json`) or a
        URL (`url`). We handle both.
        """
        try:
            import requests
        except ImportError:
            logger.error("requests not installed; T2I cannot run")
            return []

        n = n or self.t2i_n
        size = size or self.t2i_size

        payload: Dict[str, Any] = {
            "model": self.t2i_model,
            "prompt": prompt,
            "n": n,
            "image_size": size,
            "batch_size": n,
        }
        if negative:
            payload["negative_prompt"] = negative

        headers = {
            "Authorization": f"Bearer {self._t2i_api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.retry_count):
            try:
                r = requests.post(
                    self.t2i_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"T2I HTTP {r.status_code}: {r.text[:200]}")
                data = r.json()
                images = self._parse_t2i_response(data)
                time.sleep(self.rate_limit_delay)
                if images:
                    return images
                raise RuntimeError(f"T2I empty response: {str(data)[:200]}")
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(f"T2I attempt {attempt+1}/{self.retry_count} failed: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))
        logger.error(f"T2I failed after {self.retry_count} retries: {last_err}")
        return []

    async def async_call_t2i(self, prompt: str, negative: str = "", n: Optional[int] = None) -> List[bytes]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_t2i, prompt, negative, n)

    @staticmethod
    def _parse_t2i_response(data: Dict[str, Any]) -> List[bytes]:
        out: List[bytes] = []
        items = data.get("data") or data.get("images") or []
        try:
            import requests
        except ImportError:
            requests = None  # type: ignore

        for item in items:
            if not isinstance(item, dict):
                continue
            if "b64_json" in item and item["b64_json"]:
                try:
                    out.append(base64.b64decode(item["b64_json"]))
                    continue
                except Exception as e:
                    logger.warning(f"Failed to decode b64_json: {e}")
            url = item.get("url") or item.get("image_url")
            if url and requests is not None:
                try:
                    rr = requests.get(url, timeout=30)
                    rr.raise_for_status()
                    out.append(rr.content)
                except Exception as e:
                    logger.warning(f"Failed to download T2I url {url}: {e}")
        return out

    # ---------------- Structured VQA ----------------
    def call_vqa_structured(
        self,
        image_path: str,
        qc_prompt: str,
    ) -> Dict[str, Any]:
        """
        Send (image, qc_prompt) and parse the model's strict-JSON answer.

        Returns dict like::

            {
                "checks": [{"name": "...", "answer": True, "confidence": 0.92, "rationale": "..."}, ...],
                "raw": "..."
            }

        On parse failure, `checks` will be an empty list and `raw` will contain the raw
        response for diagnosis.
        """
        raw = self.call_vlm(image_path, qc_prompt)
        checks = self._extract_checks(raw)
        return {"checks": checks, "raw": raw}

    @staticmethod
    def _extract_checks(text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        # 1) try direct JSON parse
        for candidate in _iter_json_candidates(text):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list):
                return [c for c in obj if isinstance(c, dict)]
            if isinstance(obj, dict):
                if isinstance(obj.get("checks"), list):
                    return [c for c in obj["checks"] if isinstance(c, dict)]
                # tolerate {"name":..., "answer":..., ...} as single check
                if {"name", "answer"}.issubset(obj.keys()):
                    return [obj]
        return []


def _iter_json_candidates(text: str):
    """Yield JSON-ish substrings: full text, fenced ```json blocks, then [..]/{..} substring."""
    yield text.strip()
    # fenced
    for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        yield m.group(1).strip()
    # bracket scan: first [..] or {..}
    for opener, closer in (("[", "]"), ("{", "}")):
        i = text.find(opener)
        if i == -1:
            continue
        depth = 0
        for j in range(i, len(text)):
            ch = text[j]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
