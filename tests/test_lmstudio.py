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
    def test_localhost_port_1234_returns_lmstudio(self):
        assert llm_core._detect_provider("http://localhost:1234/v1/chat/completions") == "lmstudio"

    def test_127_port_1234_returns_lmstudio(self):
        assert llm_core._detect_provider("http://127.0.0.1:1234/v1/chat/completions") == "lmstudio"

    def test_lan_ip_port_1234_returns_lmstudio(self):
        assert llm_core._detect_provider("http://192.168.1.10:1234/v1/chat/completions") == "lmstudio"

    def test_port_1234_no_path_returns_lmstudio(self):
        assert llm_core._detect_provider("http://localhost:1234") == "lmstudio"

    def test_port_11234_does_not_return_lmstudio(self):
        # Substring false-positive guard: 11234 contains "1234" but is not port 1234.
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
