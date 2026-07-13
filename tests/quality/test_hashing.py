"""Tests for i2vcompbench.quality.hashing module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import hashlib

import pytest

from i2vcompbench.quality.hashing import (
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    stable_order_key,
)


class TestBytesSha256:
    """bytes_sha256 produces deterministic results."""

    def test_deterministic(self):
        result1 = bytes_sha256(b"hello")
        result2 = bytes_sha256(b"hello")
        assert result1 == result2

    def test_known_value(self):
        # SHA-256 of b"hello" is well-known
        expected = hashlib.sha256(b"hello").hexdigest()
        assert bytes_sha256(b"hello") == expected

    def test_different_inputs(self):
        assert bytes_sha256(b"hello") != bytes_sha256(b"world")

    def test_empty_bytes(self):
        result = bytes_sha256(b"")
        assert len(result) == 64  # SHA-256 hex is 64 chars


class TestCanonicalJsonSha256:
    """canonical_json_sha256 normalizes key order."""

    def test_key_order_invariant(self):
        hash1 = canonical_json_sha256({"b": 2, "a": 1})
        hash2 = canonical_json_sha256({"a": 1, "b": 2})
        assert hash1 == hash2

    def test_nested_key_order(self):
        hash1 = canonical_json_sha256({"x": {"b": 2, "a": 1}})
        hash2 = canonical_json_sha256({"x": {"a": 1, "b": 2}})
        assert hash1 == hash2

    def test_different_values_different_hash(self):
        hash1 = canonical_json_sha256({"a": 1})
        hash2 = canonical_json_sha256({"a": 2})
        assert hash1 != hash2

    def test_deterministic(self):
        obj = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
        assert canonical_json_sha256(obj) == canonical_json_sha256(obj)


class TestStableOrderKey:
    """stable_order_key is deterministic and varies by inputs."""

    def test_deterministic(self):
        key1 = stable_order_key(42, "question_001")
        key2 = stable_order_key(42, "question_001")
        assert key1 == key2

    def test_different_qid_different_key(self):
        key1 = stable_order_key(42, "question_001")
        key2 = stable_order_key(42, "question_002")
        assert key1 != key2

    def test_different_seed_different_key(self):
        key1 = stable_order_key(42, "question_001")
        key2 = stable_order_key(99, "question_001")
        assert key1 != key2

    def test_output_format(self):
        key = stable_order_key(1, "test")
        assert len(key) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in key)


class TestFileSha256:
    """file_sha256 works correctly on temp files."""

    def test_basic_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        result = file_sha256(f)
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert result == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        result = file_sha256(f)
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_binary_file(self, tmp_path):
        f = tmp_path / "binary.bin"
        data = bytes(range(256)) * 300  # ~76KB to test chunked reading
        f.write_bytes(data)
        result = file_sha256(f)
        expected = hashlib.sha256(data).hexdigest()
        assert result == expected
