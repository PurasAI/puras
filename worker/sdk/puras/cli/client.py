"""Thin httpx wrapper for the public Puras API, authenticated with a Bearer key."""

from __future__ import annotations

from typing import Any

import httpx


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def _detail(r: httpx.Response) -> str:
    try:
        j = r.json()
        if isinstance(j, dict):
            return str(j.get("detail") or j.get("message") or j)
        return str(j)
    except ValueError:
        return r.text or r.reason_phrase


class ApiClient:
    def __init__(self, api_base: str, api_key: str | None, timeout: float = 60.0):
        self.api_base = api_base.rstrip("/")
        self._key = api_key
        self._c = httpx.Client(timeout=timeout, follow_redirects=True)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"} if self._key else {}

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: Any = None,
        files: dict | None = None,
        data: dict | None = None,
    ) -> Any:
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        r = self._c.request(
            method,
            url,
            params=params,
            json=json_body,
            files=files,
            data=data,
            headers=self._headers(),
        )
        if r.status_code >= 400:
            raise ApiError(r.status_code, _detail(r))
        if r.status_code == 204 or not r.content:
            return None
        if "application/json" in r.headers.get("content-type", ""):
            return r.json()
        return r.text

    def get(self, path: str, **kw: Any) -> Any:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> Any:
        return self.request("POST", path, **kw)

    def put(self, path: str, **kw: Any) -> Any:
        return self.request("PUT", path, **kw)

    def delete(self, path: str, **kw: Any) -> Any:
        return self.request("DELETE", path, **kw)

    def close(self) -> None:
        self._c.close()
