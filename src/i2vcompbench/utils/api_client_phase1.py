"""
SiliconFlow API client for VLM (image analysis) and LLM (text analysis) calls.
Uses OpenAI-compatible SDK with SiliconFlow's base_url.
"""

import asyncio
import base64
import os
import time
from io import BytesIO
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI, OpenAI
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


MAX_IMAGE_LONG_EDGE = 2048


def _resolve_model(yaml_default: str, env_key: str) -> str:
    """优先读环境变量（可由 .env 提供），未设则回退到 yaml 默认值。"""
    v = os.environ.get(env_key)
    return v.strip() if v and v.strip() else yaml_default


def _resize_image_if_needed(image_path: str) -> str:
    """
    Read image, resize if long edge > MAX_IMAGE_LONG_EDGE,
    return base64-encoded JPEG string.
    """
    img = Image.open(image_path)
    w, h = img.size
    long_edge = max(w, h)

    if long_edge > MAX_IMAGE_LONG_EDGE:
        scale = MAX_IMAGE_LONG_EDGE / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"Resized image {image_path}: {w}x{h} -> {new_w}x{new_h}")

    # Convert to RGB if needed (e.g., RGBA or palette images)
    if img.mode not in ("RGB",):
        img = img.convert("RGB")

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64_str


class SiliconFlowClient:
    """Unified client for SiliconFlow VLM and LLM API calls."""

    def __init__(self, config: dict):
        api_cfg = config["api"]

        api_key = os.environ.get(api_cfg["api_key_env"], "")
        if not api_key:
            logger.warning(
                f"Environment variable {api_cfg['api_key_env']} not set. "
                "API calls will fail."
            )

        self.base_url = api_cfg["base_url"]

        # Sync client (for single calls)
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

        # Async client (for batch concurrent calls)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)

        # VLM config
        vlm_cfg = api_cfg["vlm"]
        self.vlm_model = _resolve_model(vlm_cfg["model"], "VLM_MODEL")
        self.vlm_max_tokens = vlm_cfg.get("max_tokens", 2048)
        self.vlm_temperature = vlm_cfg.get("temperature", 0.0)

        # LLM config
        llm_cfg = api_cfg["llm"]
        self.llm_model = _resolve_model(llm_cfg["model"], "LLM_MODEL")
        self.llm_max_tokens = llm_cfg.get("max_tokens", 1500)
        self.llm_temperature = llm_cfg.get("temperature", 0.0)

        # Batch / rate limit config
        self.batch_size = api_cfg.get("batch_size", 5)
        self.retry_count = api_cfg.get("retry_count", 3)
        self.retry_delay = api_cfg.get("retry_delay", 2)
        self.timeout = api_cfg.get("timeout", 120)
        self.rate_limit_delay = api_cfg.get("rate_limit_delay", 0.5)

        logger.info(
            f"SiliconFlowClient initialized: VLM={self.vlm_model}, LLM={self.llm_model}, "
            f"batch_size={self.batch_size}"
        )

    def call_vlm(self, image_path: str, prompt_text: str) -> str:
        """
        Synchronous VLM call: send image + text prompt, return response text.
        """
        try:
            b64_img = _resize_image_if_needed(image_path)
        except Exception as e:
            logger.error(f"Failed to read/resize image {image_path}: {e}")
            return ""

        @retry(
            stop=stop_after_attempt(self.retry_count),
            wait=wait_exponential(multiplier=self.retry_delay, min=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call():
            response = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_img}"
                                },
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                max_tokens=self.vlm_max_tokens,
                temperature=self.vlm_temperature,
                timeout=self.timeout,
            )
            return response.choices[0].message.content or ""

        try:
            result = _call()
            time.sleep(self.rate_limit_delay)
            return result
        except Exception as e:
            logger.error(f"VLM call failed after retries for {image_path}: {e}")
            return ""

    def call_llm(self, prompt_text: str) -> str:
        """
        Synchronous LLM call: send text prompt, return response text.
        """

        @retry(
            stop=stop_after_attempt(self.retry_count),
            wait=wait_exponential(multiplier=self.retry_delay, min=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call():
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature,
                timeout=self.timeout,
            )
            return response.choices[0].message.content or ""

        try:
            result = _call()
            time.sleep(self.rate_limit_delay)
            return result
        except Exception as e:
            logger.error(f"LLM call failed after retries: {e}")
            return ""

    async def async_call_vlm(self, image_path: str, prompt_text: str) -> str:
        """Async VLM call for concurrent batch processing."""
        try:
            b64_img = _resize_image_if_needed(image_path)
        except Exception as e:
            logger.error(f"Failed to read/resize image {image_path}: {e}")
            return ""

        for attempt in range(self.retry_count):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_img}"
                                    },
                                },
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ],
                    max_tokens=self.vlm_max_tokens,
                    temperature=self.vlm_temperature,
                    timeout=self.timeout,
                )
                await asyncio.sleep(self.rate_limit_delay)
                content = response.choices[0].message.content or ""
                if not content and response.choices[0].finish_reason == "length":
                    logger.warning(
                        f"VLM response content empty due to max_tokens exhausted by reasoning "
                        f"for {image_path}. Consider increasing max_tokens."
                    )
                return content
            except Exception as e:
                err_msg = str(e)
                is_rate_limit = "429" in err_msg or "rate limiting" in err_msg.lower()
                logger.warning(
                    f"Async VLM attempt {attempt + 1}/{self.retry_count} "
                    f"failed for {image_path}: {e}"
                )
                if attempt < self.retry_count - 1:
                    if is_rate_limit:
                        # Longer backoff for rate limit errors
                        wait_time = 30 * (2 ** attempt)
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))

        logger.error(f"Async VLM call failed after {self.retry_count} retries: {image_path}")
        return ""

    async def async_call_llm(self, prompt_text: str) -> str:
        """Async LLM call for concurrent batch processing."""
        for attempt in range(self.retry_count):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=self.llm_max_tokens,
                    temperature=self.llm_temperature,
                    timeout=self.timeout,
                )
                await asyncio.sleep(self.rate_limit_delay)
                content = response.choices[0].message.content or ""
                if not content and response.choices[0].finish_reason == "length":
                    logger.warning(
                        "LLM response content empty due to max_tokens exhausted by reasoning. "
                        "Consider increasing max_tokens."
                    )
                return content
            except Exception as e:
                err_msg = str(e)
                is_rate_limit = "429" in err_msg or "rate limiting" in err_msg.lower()
                logger.warning(
                    f"Async LLM attempt {attempt + 1}/{self.retry_count} failed: {e}"
                )
                if attempt < self.retry_count - 1:
                    if is_rate_limit:
                        wait_time = 30 * (2 ** attempt)
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))

        logger.error(f"Async LLM call failed after {self.retry_count} retries")
        return ""
