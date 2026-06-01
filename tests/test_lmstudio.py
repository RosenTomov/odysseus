"""Tests for LM Studio provider detection, labeling, discovery, and streaming."""
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ── Stub heavy optional deps that may not be installed ──
for _mod in ("sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext",
             "sqlalchemy.ext.declarative", "sqlalchemy.ext.hybrid",
             "sqlalchemy.sql", "sqlalchemy.sql.expression",
             "sqlalchemy.sql.sqltypes", "sqlalchemy.types",
             "bcrypt", "pyotp"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

if "src.database" not in sys.modules:
    _db = types.ModuleType("src.database")
    _db.SessionLocal = MagicMock()
    _db.ModelEndpoint = MagicMock()
    sys.modules["src.database"] = _db

from src import llm_core
from src.model_discovery import ModelDiscovery


# ════════════════════════════════════════════════════════════
# 1. _detect_provider — LM Studio
# ════════════════════════════════════════════════════════════

class TestDetectProviderLmStudio:
    """Port 1234 is necessary but no longer sufficient: the result is confirmed
    by the native-API fingerprint so other servers on 1234 aren't misdetected."""

    def test_localhost_port_1234_fingerprint_confirms(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_fingerprint_is_lmstudio", lambda host, port: True)
        assert llm_core._detect_provider("http://localhost:1234/v1/chat/completions") == "lmstudio"

    def test_127_port_1234_fingerprint_confirms(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_fingerprint_is_lmstudio", lambda host, port: True)
        assert llm_core._detect_provider("http://127.0.0.1:1234/v1/chat/completions") == "lmstudio"

    def test_lan_ip_port_1234_fingerprint_confirms(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_fingerprint_is_lmstudio", lambda host, port: True)
        assert llm_core._detect_provider("http://192.168.1.10:1234/v1/chat/completions") == "lmstudio"

    def test_port_1234_no_path_fingerprint_confirms(self, monkeypatch):
        monkeypatch.setattr(llm_core, "_fingerprint_is_lmstudio", lambda host, port: True)
        assert llm_core._detect_provider("http://localhost:1234") == "lmstudio"

    def test_port_1234_non_lmstudio_server_not_misdetected(self, monkeypatch):
        # vLLM / llama.cpp / a proxy on 1234: the fingerprint fails, so the
        # result must NOT be lmstudio — otherwise stream_options is silently
        # dropped and token-usage stats break for that server.
        monkeypatch.setattr(llm_core, "_fingerprint_is_lmstudio", lambda host, port: False)
        assert llm_core._detect_provider("http://localhost:1234/v1/chat/completions") == "openai"

    def test_port_11234_does_not_return_lmstudio(self):
        # Substring false-positive guard: 11234 contains "1234" but is not port
        # 1234, so the probe never runs.
        result = llm_core._detect_provider("http://localhost:11234/v1/chat/completions")
        assert result != "lmstudio"

    def test_port_12340_does_not_return_lmstudio(self):
        result = llm_core._detect_provider("http://localhost:12340/v1/chat/completions")
        assert result != "lmstudio"


# ════════════════════════════════════════════════════════════
# 2. _detect_provider — other providers still work
# ════════════════════════════════════════════════════════════

class TestDetectProviderOtherProviders:
    @pytest.mark.parametrize("url,expected", [
        ("http://localhost:11434/api/chat", "ollama"),
        ("https://ollama.com/api/chat", "ollama"),
        ("https://api.anthropic.com/v1/messages", "anthropic"),
        ("https://openrouter.ai/api/v1/chat/completions", "openrouter"),
        ("https://api.groq.com/openai/v1/chat/completions", "groq"),
        ("https://api.openai.com/v1/chat/completions", "openai"),
    ])
    def test_provider_detected_correctly(self, url, expected):
        assert llm_core._detect_provider(url) == expected

    def test_empty_string_returns_openai(self):
        assert llm_core._detect_provider("") == "openai"

    def test_none_like_empty_returns_openai(self):
        # _detect_provider coerces None-equivalent empty string
        assert llm_core._detect_provider("") == "openai"


# ════════════════════════════════════════════════════════════
# 3. _provider_label — LM Studio
# ════════════════════════════════════════════════════════════

class TestProviderLabelLmStudio:
    def test_localhost_port_1234_label(self):
        assert llm_core._provider_label("http://localhost:1234/v1/chat/completions") == "LM Studio"

    def test_lan_ip_port_1234_label(self):
        assert llm_core._provider_label("http://192.168.1.10:1234/v1/chat/completions") == "LM Studio"

    def test_port_1234_without_path_label(self):
        assert llm_core._provider_label("http://localhost:1234") == "LM Studio"


class TestProviderLabelOtherProviders:
    @pytest.mark.parametrize("url,expected", [
        ("https://api.anthropic.com/v1/messages", "Anthropic"),
        ("https://ollama.com/api/chat", "Ollama Cloud"),
        ("https://api.openai.com/v1/chat/completions", "OpenAI"),
        ("https://openrouter.ai/api/v1/chat/completions", "OpenRouter"),
        ("https://api.groq.com/openai/v1/chat/completions", "Groq"),
    ])
    def test_label_matches_provider(self, url, expected):
        assert llm_core._provider_label(url) == expected


# ════════════════════════════════════════════════════════════
# 4. ModelDiscovery — ports list includes 1234
# ════════════════════════════════════════════════════════════

class TestModelDiscoveryPorts:
    def test_discover_models_scans_port_1234(self, monkeypatch):
        """discover_models must include port 1234 among the scan targets."""
        discovery = ModelDiscovery(default_host="localhost")
        scanned_ports = []

        def fake_check_port(host, port):
            scanned_ports.append(port)
            return None

        monkeypatch.setattr(discovery, "_check_port", fake_check_port)
        monkeypatch.setattr(
            "src.model_discovery.discover_tailscale_hosts",
            lambda: [],
        )

        discovery.discover_models()
        assert 1234 in scanned_ports


# ════════════════════════════════════════════════════════════
# 4b. _fingerprint_provider — native API identification
# ════════════════════════════════════════════════════════════

class _FakeResponse:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.is_success = ok

    def json(self):
        return self._payload


class TestFingerprintProvider:
    LMSTUDIO_NATIVE = {
        "models": [
            {"type": "llm", "key": "qwen3.6-27b", "architecture": "qwen35",
             "quantization": {"name": "Q5_K_XL"}, "format": "gguf"},
        ]
    }

    def test_lmstudio_native_format_detected(self, monkeypatch):
        discovery = ModelDiscovery(default_host="localhost")
        monkeypatch.setattr(
            "src.model_discovery.httpx.get",
            lambda url, timeout=None: _FakeResponse(self.LMSTUDIO_NATIVE),
        )
        assert discovery._fingerprint_provider("localhost", 1234) == "lmstudio"

    def test_lmstudio_detected_on_nonstandard_port(self, monkeypatch):
        discovery = ModelDiscovery(default_host="localhost")
        monkeypatch.setattr(
            "src.model_discovery.httpx.get",
            lambda url, timeout=None: _FakeResponse(self.LMSTUDIO_NATIVE),
        )
        assert discovery._fingerprint_provider("localhost", 8080) == "lmstudio"

    def test_openai_compatible_server_not_lmstudio(self, monkeypatch):
        discovery = ModelDiscovery(default_host="localhost")
        monkeypatch.setattr(
            "src.model_discovery.httpx.get",
            lambda url, timeout=None: _FakeResponse({"data": [{"id": "gpt-4o"}]}, ok=False),
        )
        assert discovery._fingerprint_provider("localhost", 8000) is None

    def test_ollama_tags_shape_not_lmstudio(self, monkeypatch):
        discovery = ModelDiscovery(default_host="localhost")
        ollama_shape = {"models": [{"name": "llama3", "modified_at": "x", "size": 1}]}
        monkeypatch.setattr(
            "src.model_discovery.httpx.get",
            lambda url, timeout=None: _FakeResponse(ollama_shape),
        )
        assert discovery._fingerprint_provider("localhost", 11434) is None

    def test_unreachable_returns_none(self, monkeypatch):
        discovery = ModelDiscovery(default_host="localhost")
        def boom(url, timeout=None):
            raise OSError("connection refused")
        monkeypatch.setattr("src.model_discovery.httpx.get", boom)
        assert discovery._fingerprint_provider("localhost", 1234) is None

    def test_check_port_attaches_provider(self, monkeypatch):
        discovery = ModelDiscovery(default_host="localhost")

        def fake_get(url, timeout=None):
            if url.endswith("/api/v1/models"):
                return _FakeResponse(self.LMSTUDIO_NATIVE)
            return _FakeResponse({"data": [{"id": "qwen3.6-27b"}]})

        monkeypatch.setattr("src.model_discovery.httpx.get", fake_get)
        result = discovery._check_port("localhost", 1234)
        assert result is not None
        assert result["provider"] == "lmstudio"
        assert result["models"] == ["qwen3.6-27b"]


# ════════════════════════════════════════════════════════════
# 4c. _is_lmstudio_models_payload — shared shape check
# ════════════════════════════════════════════════════════════

class TestIsLmStudioModelsPayload:
    def test_lmstudio_shape_true(self):
        payload = {"models": [{"key": "qwen3", "architecture": "qwen35"}]}
        assert llm_core._is_lmstudio_models_payload(payload) is True

    @pytest.mark.parametrize("payload", [
        {},
        {"models": []},
        {"models": [{"id": "gpt-4o"}]},                       # OpenAI-compatible shape
        {"models": [{"name": "llama3", "modified_at": "x"}]},  # Ollama /tags shape
        {"data": [{"id": "gpt-4o"}]},                          # no "models" key
    ])
    def test_non_lmstudio_shapes_false(self, payload):
        assert llm_core._is_lmstudio_models_payload(payload) is False


# ════════════════════════════════════════════════════════════
# 4d. _fingerprint_is_lmstudio — cached native-API probe
# ════════════════════════════════════════════════════════════

class TestFingerprintIsLmStudio:
    LMSTUDIO_NATIVE = {"models": [{"key": "qwen3", "architecture": "qwen35"}]}

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        llm_core._provider_fingerprint_cache.clear()
        yield
        llm_core._provider_fingerprint_cache.clear()

    def test_positive_shape_returns_true(self, monkeypatch):
        monkeypatch.setattr(llm_core.httpx, "get",
                            lambda url, timeout=None: _FakeResponse(self.LMSTUDIO_NATIVE))
        assert llm_core._fingerprint_is_lmstudio("localhost", 1234) is True

    def test_responding_non_lmstudio_returns_false(self, monkeypatch):
        monkeypatch.setattr(llm_core.httpx, "get",
                            lambda url, timeout=None: _FakeResponse({"data": [{"id": "x"}]}))
        assert llm_core._fingerprint_is_lmstudio("localhost", 1234) is False

    def test_probe_error_returns_false_and_not_cached(self, monkeypatch):
        def boom(url, timeout=None):
            raise OSError("connection refused")
        monkeypatch.setattr(llm_core.httpx, "get", boom)
        assert llm_core._fingerprint_is_lmstudio("localhost", 1234) is False
        # A transient error must not be cached, so the next call re-probes.
        assert ("localhost", 1234) not in llm_core._provider_fingerprint_cache

    def test_result_is_cached_within_ttl(self, monkeypatch):
        calls = {"n": 0}

        def counting_get(url, timeout=None):
            calls["n"] += 1
            return _FakeResponse(self.LMSTUDIO_NATIVE)

        monkeypatch.setattr(llm_core.httpx, "get", counting_get)
        assert llm_core._fingerprint_is_lmstudio("localhost", 1234) is True
        assert llm_core._fingerprint_is_lmstudio("localhost", 1234) is True
        assert calls["n"] == 1  # second call served from cache, no re-probe


# ════════════════════════════════════════════════════════════
# 5. _get_hosts — LM_STUDIO_URL env var
# ════════════════════════════════════════════════════════════

class TestGetHostsLmStudioUrl:
    def test_lm_studio_url_adds_host_default_branch(self, monkeypatch):
        """LM_STUDIO_URL hostname must appear in hosts when Tailscale is absent."""
        monkeypatch.delenv("LLM_HOSTS", raising=False)
        monkeypatch.setenv("LM_STUDIO_URL", "http://my-lm-box:1234")
        monkeypatch.setattr(
            "src.model_discovery.discover_tailscale_hosts",
            lambda: [],
        )
        discovery = ModelDiscovery(default_host="localhost")
        hosts = discovery._get_hosts()
        assert "my-lm-box" in hosts

    def test_lm_studio_url_adds_host_tailscale_branch(self, monkeypatch):
        """LM_STUDIO_URL hostname must also appear when Tailscale hosts are present."""
        monkeypatch.delenv("LLM_HOSTS", raising=False)
        monkeypatch.setenv("LM_STUDIO_URL", "http://my-lm-box:1234")
        monkeypatch.setattr(
            "src.model_discovery.discover_tailscale_hosts",
            lambda: ["100.64.0.1"],
        )
        discovery = ModelDiscovery(default_host="localhost")
        hosts = discovery._get_hosts()
        assert "my-lm-box" in hosts

    def test_lm_studio_url_adds_host_llm_hosts_branch(self, monkeypatch):
        """LM_STUDIO_URL hostname must also appear when LLM_HOSTS is set."""
        monkeypatch.setenv("LLM_HOSTS", "10.0.0.5")
        monkeypatch.setenv("LM_STUDIO_URL", "http://my-lm-box:1234")
        discovery = ModelDiscovery(default_host="localhost")
        hosts = discovery._get_hosts()
        assert "my-lm-box" in hosts

    def test_lm_studio_url_no_duplicate(self, monkeypatch):
        """If the hostname is already in the list it should not be added twice."""
        monkeypatch.delenv("LLM_HOSTS", raising=False)
        monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:1234")
        monkeypatch.setattr(
            "src.model_discovery.discover_tailscale_hosts",
            lambda: [],
        )
        discovery = ModelDiscovery(default_host="localhost")
        hosts = discovery._get_hosts()
        assert hosts.count("localhost") == 1

    def test_lm_studio_url_not_set_no_extra_host(self, monkeypatch):
        """When LM_STUDIO_URL is absent, no phantom host is added."""
        monkeypatch.delenv("LLM_HOSTS", raising=False)
        monkeypatch.delenv("LM_STUDIO_URL", raising=False)
        monkeypatch.setattr(
            "src.model_discovery.discover_tailscale_hosts",
            lambda: [],
        )
        discovery = ModelDiscovery(default_host="localhost")
        hosts = discovery._get_hosts()
        # Only localhost + host.docker.internal expected
        assert "my-lm-box" not in hosts


# ════════════════════════════════════════════════════════════
# 6. stream_llm — stream_options excluded for lmstudio
# ════════════════════════════════════════════════════════════

def _make_fake_client(captured: dict):
    """Return a fake AsyncClient whose .stream() is a proper async context manager."""
    from contextlib import asynccontextmanager

    class FakeClient:
        is_closed = False

        @asynccontextmanager
        async def stream(self, method, url, **kwargs):
            captured["payload"] = kwargs.get("json", {})

            class FakeResponse:
                status_code = 200

                async def aiter_lines(self):
                    yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
                    yield "data: [DONE]"

                async def aread(self):
                    return b""

            yield FakeResponse()

    return FakeClient()


class TestStreamOptionsExcluded:
    @pytest.mark.asyncio
    async def test_stream_options_absent_for_lmstudio(self, monkeypatch):
        """stream_options must NOT be included in the payload sent to LM Studio."""
        captured = {}
        monkeypatch.setattr(llm_core, "_get_http_client", lambda: _make_fake_client(captured))
        monkeypatch.setattr(llm_core, "_fingerprint_is_lmstudio", lambda host, port: True)

        chunks = []
        async for chunk in llm_core.stream_llm(
            "http://localhost:1234/v1/chat/completions",
            "lmstudio-model",
            [{"role": "user", "content": "hi"}],
        ):
            chunks.append(chunk)

        assert "stream_options" not in captured.get("payload", {}), (
            f"stream_options was unexpectedly present in payload: {captured.get('payload')}"
        )

    @pytest.mark.asyncio
    async def test_stream_options_present_for_openai(self, monkeypatch):
        """stream_options SHOULD be included for OpenAI-compatible endpoints."""
        captured = {}
        monkeypatch.setattr(llm_core, "_get_http_client", lambda: _make_fake_client(captured))

        chunks = []
        async for chunk in llm_core.stream_llm(
            "http://localhost:8080/v1/chat/completions",
            "some-model",
            [{"role": "user", "content": "hi"}],
        ):
            chunks.append(chunk)

        assert "stream_options" in captured.get("payload", {}), (
            "stream_options should be present for non-excluded providers"
        )


# ════════════════════════════════════════════════════════════
# 7. build_chat_url / build_models_url — LM Studio routing
# ════════════════════════════════════════════════════════════

class TestEndpointResolverLmStudio:
    def test_build_chat_url_lmstudio(self, monkeypatch):
        import src.endpoint_resolver as er
        # Skip DNS / Tailscale resolution
        monkeypatch.setattr(er, "resolve_url", lambda url: url)
        result = er.build_chat_url("http://localhost:1234/v1")
        assert result == "http://localhost:1234/v1/chat/completions"

    def test_build_models_url_lmstudio(self, monkeypatch):
        import src.endpoint_resolver as er
        monkeypatch.setattr(er, "resolve_url", lambda url: url)
        result = er.build_models_url("http://localhost:1234/v1")
        assert result == "http://localhost:1234/v1/models"

    def test_build_chat_url_lan_lmstudio(self, monkeypatch):
        import src.endpoint_resolver as er
        monkeypatch.setattr(er, "resolve_url", lambda url: url)
        result = er.build_chat_url("http://192.168.1.5:1234/v1")
        assert result == "http://192.168.1.5:1234/v1/chat/completions"
