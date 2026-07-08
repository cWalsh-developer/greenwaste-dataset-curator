from pathlib import Path

import requests

from greenwaste_dataset_curator.commons import CommonsCollector


class FakeResponse:
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self) -> None:
        self.headers = {}
        self.responses = [FakeResponse(429, retry_after="0"), FakeResponse(200)]
        self.calls = 0

    def get(self, *args, **kwargs) -> FakeResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_get_with_backoff_retries_http_429(tmp_path: Path) -> None:
    collector = CommonsCollector(output_dir=tmp_path, max_retries=2, backoff_seconds=0)
    fake_session = FakeSession()
    collector.session = fake_session  # type: ignore[assignment]

    response = collector.get_with_backoff("https://example.com")

    assert response.status_code == 200
    assert fake_session.calls == 2
