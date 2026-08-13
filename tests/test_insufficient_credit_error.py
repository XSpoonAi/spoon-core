import pytest

from spoon_ai.llm.errors import InsufficientCreditsError, RateLimitError
from spoon_ai.llm.providers.openai_compatible_provider import OpenAICompatibleProvider


class FakeAPIError(Exception):
    def __init__(self, status_code, body, message=None):
        super().__init__(message or str(body))
        self.status_code = status_code
        self.body = body


@pytest.mark.asyncio
async def test_billing_402_is_not_rate_limit():
    provider = OpenAICompatibleProvider()
    error = FakeAPIError(
        402,
        {
            "error": {
                "type": "insufficient_quota",
                "code": "insufficient_funds",
                "message": "insufficient funds",
            }
        },
    )

    with pytest.raises(InsufficientCreditsError) as raised:
        await provider._handle_error(error)

    assert raised.value.status_code == 402
    assert raised.value.error_code == "insufficient_funds"
    assert not isinstance(raised.value, RateLimitError)


@pytest.mark.asyncio
async def test_provider_rate_limit_stays_rate_limit():
    provider = OpenAICompatibleProvider()
    error = FakeAPIError(
        429,
        {
            "error": {
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
                "message": "too many requests",
            }
        },
    )

    with pytest.raises(RateLimitError):
        await provider._handle_error(error)
