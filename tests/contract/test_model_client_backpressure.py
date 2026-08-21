from __future__ import annotations

import anthropic
import httpx
import pytest

from env_mock_agent.models import ModelClientBackpressureError, anthropic_client, ark_client


def _anthropic_error(status_code: int, *, retry_after: str) -> Exception:
    response = httpx.Response(
        status_code,
        headers={"retry-after": retry_after},
        request=httpx.Request("POST", "https://api.anthropic.test/v1/messages"),
    )
    error_type: type[Exception] = (
        anthropic.RateLimitError if status_code == 429 else anthropic.InternalServerError
    )
    return error_type("typed provider error", response=response, body=None)


class _RaisingStream:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aenter__(self) -> _RaisingStream:
        raise self.error

    async def __aexit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        return None


class _Messages:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.kwargs: dict[str, object] | None = None

    def stream(self, **kwargs: object) -> _RaisingStream:
        self.kwargs = kwargs
        return _RaisingStream(self.error)


class _AnthropicClient:
    def __init__(self, error: Exception) -> None:
        self.messages = _Messages(error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "retry_after", "expected_kind", "expected_seconds"),
    (
        (429, "0", "RATE_LIMITED", 1),
        (529, "999999", "OVERLOADED", 86_400),
    ),
)
async def test_anthropic_client_converts_only_typed_backpressure(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    retry_after: str,
    expected_kind: str,
    expected_seconds: int,
) -> None:
    fake = _AnthropicClient(_anthropic_error(status_code, retry_after=retry_after))
    constructor_values: list[int] = []

    def client_factory(*, max_retries: int) -> _AnthropicClient:
        constructor_values.append(max_retries)
        return fake

    monkeypatch.setattr(
        anthropic_client.anthropic,
        "AsyncAnthropic",
        client_factory,
    )
    client = anthropic_client.AnthropicStructuredClient(model="model://exact")

    with pytest.raises(ModelClientBackpressureError) as captured:
        await client.generate_json(
            system="system",
            prompt="prompt",
            schema={"type": "object"},
            max_output_tokens=321,
        )

    assert constructor_values == [0]
    assert captured.value.kind == expected_kind
    assert captured.value.retry_after_seconds == expected_seconds
    assert fake.messages.kwargs is not None
    assert fake.messages.kwargs["model"] == "model://exact"
    assert fake.messages.kwargs["max_tokens"] == 321


@pytest.mark.asyncio
async def test_anthropic_non_529_server_error_is_not_backpressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _anthropic_error(500, retry_after="5")
    fake = _AnthropicClient(error)
    monkeypatch.setattr(
        anthropic_client.anthropic,
        "AsyncAnthropic",
        lambda **_kwargs: fake,
    )
    client = anthropic_client.AnthropicStructuredClient()

    with pytest.raises(anthropic.InternalServerError) as captured:
        await client.generate_json(
            system="system",
            prompt="prompt",
            schema={"type": "object"},
        )

    assert captured.value is error


class _ArkClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.json_payload: dict[str, object] | None = None

    async def __aenter__(self) -> _ArkClient:
        return self

    async def __aexit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        return None

    async def post(
        self,
        _url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
    ) -> httpx.Response:
        assert headers["Authorization"] == "Bearer test-key"
        self.json_payload = json
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "retry_after", "expected_kind", "expected_seconds"),
    (
        (429, "7", "RATE_LIMITED", 7),
        (529, "invalid", "OVERLOADED", 60),
    ),
)
async def test_ark_client_converts_structured_backpressure_and_preserves_cap(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    retry_after: str,
    expected_kind: str,
    expected_seconds: int,
) -> None:
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    fake = _ArkClient(
        httpx.Response(
            status_code,
            headers={"retry-after": retry_after},
            request=httpx.Request("POST", "https://ark.test/chat/completions"),
        )
    )

    def factory(**_kwargs: object) -> _ArkClient:
        return fake

    monkeypatch.setattr(ark_client.httpx, "AsyncClient", factory)
    client = ark_client.ArkStructuredClient(
        model="model://exact",
        base_url="https://ark.test/",
    )

    with pytest.raises(ModelClientBackpressureError) as captured:
        await client.generate_json(
            system="system",
            prompt="prompt",
            schema={"type": "object"},
            max_output_tokens=456,
        )

    assert captured.value.kind == expected_kind
    assert captured.value.retry_after_seconds == expected_seconds
    assert fake.json_payload is not None
    assert fake.json_payload["model"] == "model://exact"
    assert fake.json_payload["max_tokens"] == 456
