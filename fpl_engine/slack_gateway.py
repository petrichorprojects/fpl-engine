"""Slack I/O for the lineup nag loop.

Two concrete gateways behind one protocol, so the nag logic never touches the
network directly:

- `SlackApiGateway` — talks to the Slack Web API with a bot token. Needs the bot
  in the target channel and scopes `chat:write`, plus `channels:history` /
  `groups:history` and `files:read` to see your screenshot reply.
- `DryRunGateway` — prints instead of posting. Used by `--dry-run` and tests.

Detecting the screenshot is the whole point of the read side: the gateway
returns recent messages as `nag.SlackMessage`, and `nag.find_acknowledgement`
decides whether your image counts.
"""

from __future__ import annotations

from typing import Optional, Protocol

import httpx

from .nag import SlackMessage

SLACK_API = "https://slack.com/api"


class SlackGateway(Protocol):
    def post(self, text: str, thread_ts: Optional[str] = None) -> Optional[str]:
        """Post a message; return its ts (thread root for later replies)."""
        ...

    def recent_messages(self, since_ts: float, thread_ts: Optional[str]) -> list[SlackMessage]:
        """Channel history plus any thread replies since `since_ts`."""
        ...


class SlackApiGateway:
    """Slack Web API gateway using a bot token."""

    def __init__(self, token: str, channel: str, timeout: float = 15.0):
        self.channel = channel
        self._http = httpx.Client(
            base_url=SLACK_API,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    def _call(self, method: str, **payload) -> dict:
        resp = self._http.post(f"/{method}", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Slack API {method} failed: {data.get('error')}")
        return data

    def post(self, text: str, thread_ts: Optional[str] = None) -> Optional[str]:
        payload = {"channel": self.channel, "text": text, "unfurl_links": False}
        if thread_ts:
            payload["thread_ts"] = thread_ts
        return self._call("chat.postMessage", **payload).get("ts")

    def recent_messages(self, since_ts: float, thread_ts: Optional[str]) -> list[SlackMessage]:
        raw: list[dict] = []

        hist = self._http.post(
            "/conversations.history",
            json={"channel": self.channel, "oldest": str(since_ts), "limit": 50},
        ).json()
        if hist.get("ok"):
            raw.extend(hist.get("messages", []))

        # Thread replies do not appear in channel history, so fetch them too.
        if thread_ts:
            rep = self._http.post(
                "/conversations.replies",
                json={"channel": self.channel, "ts": thread_ts, "oldest": str(since_ts)},
            ).json()
            if rep.get("ok"):
                raw.extend(rep.get("messages", []))

        return [_to_message(m) for m in raw]


class N8nRelayGateway:
    """Slack I/O routed through an n8n webhook relay.

    The relay (workflow "FPL Lineup Nag Relay") holds the Slack credential on a
    native Slack node — this repo never sees a token. We POST a small JSON body
    with a `mode` (post | history | replies) and the relay does the Slack call
    and echoes the result back.

    This mirrors the morning-briefing and banter-scorer relays: the local
    process is the brain, n8n is the authenticated Slack arm.
    """

    def __init__(self, webhook_url: str, timeout: float = 20.0):
        self.url = webhook_url
        self._http = httpx.Client(timeout=timeout)

    def _call(self, payload: dict) -> list[dict]:
        resp = self._http.post(self.url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # `respondWith: allIncomingItems` returns a list; a single Slack post
        # can come back as a bare object. Normalise to a list.
        if isinstance(data, dict):
            return [data]
        return data or []

    def post(self, text: str, thread_ts: Optional[str] = None) -> Optional[str]:
        items = self._call({"mode": "post", "text": text, "ts": thread_ts or ""})
        if not items:
            return None
        first = items[0]
        # Slack's post response nests the echoed message under `message`.
        return first.get("ts") or (first.get("message") or {}).get("ts")

    def recent_messages(self, since_ts: float, thread_ts: Optional[str]) -> list[SlackMessage]:
        raw = self._call({"mode": "history"})
        if thread_ts:
            raw += self._call({"mode": "replies", "ts": thread_ts})
        msgs = [_to_message(m) for m in raw if "ts" in m]
        return [m for m in msgs if m.ts >= since_ts]


class DryRunGateway:
    """Prints instead of posting. Records posts so tests can inspect them."""

    def __init__(self, inbound: Optional[list[SlackMessage]] = None):
        self.posted: list[tuple[str, Optional[str]]] = []
        self._inbound = inbound or []
        self._counter = 1000.0

    def post(self, text: str, thread_ts: Optional[str] = None) -> Optional[str]:
        print("\n--- SLACK (dry-run) ---")
        print(text)
        print("-----------------------")
        self.posted.append((text, thread_ts))
        self._counter += 1
        return str(self._counter)

    def recent_messages(self, since_ts: float, thread_ts: Optional[str]) -> list[SlackMessage]:
        return [m for m in self._inbound if m.ts >= since_ts]


def _to_message(m: dict) -> SlackMessage:
    """Map a raw Slack message to the fields the nag logic needs.

    A screenshot is any attached file whose mimetype starts with `image/`.
    We deliberately do not inspect the image contents — posting one is the
    ritual that proves you set your lineup; verifying it is out of scope.
    """
    files = m.get("files") or []
    has_image = any(
        str(f.get("mimetype", "")).startswith("image/")
        or str(f.get("filetype", "")).lower() in {"png", "jpg", "jpeg", "heic", "webp", "gif"}
        for f in files
    )
    return SlackMessage(
        ts=float(m.get("ts", 0.0)),
        user=str(m.get("user", "")),
        has_image=has_image,
    )
