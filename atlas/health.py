"""Keep-alive for free hosting tiers.

Free web services sleep after ~15 minutes without HTTP traffic, and a sleeping
bot means a judge messages Atlas and waits ~50 seconds for a cold start — which
reads as broken. Two pieces solve it:

  1. A tiny HTTP server so the platform sees a healthy web service at all.
  2. A periodic self-request so the service never goes idle in the first place.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

log = logging.getLogger(__name__)

# Whether the Telegram poller is still running. This matters because the health
# server lives on its own thread: when polling dies — a getUpdates Conflict from
# a redeploy overlap is the usual way — this thread carries on answering 200, the
# host sees a healthy service, and the bot silently answers nobody for hours.
# Reporting the truth here is what turns that into a restart.
_polling_stopped = threading.Event()


def mark_polling_stopped() -> None:
    _polling_stopped.set()


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/diag"):
            self._diag()
            return
        # Only ever unhealthy after polling has run and stopped; a service still
        # starting up answers 200 so the platform's first check does not fail it.
        if _polling_stopped.is_set():
            body = b"atlas unhealthy: telegram polling has stopped"
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"atlas ok")

    def _diag(self):
        """Report which market-data providers work FROM THIS HOST.

        Provider availability depends on the caller's IP: Yahoo blocks datacenter
        ranges, so a laptop and a cloud host give different answers. This can only
        be measured from where the bot actually runs.
        """
        from atlas.integrations import marketdata

        payload = {"providers": marketdata.probe("AAPL")}
        body = json.dumps(payload, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Silence per-request logging; the self-ping would flood the log."""


def start_health_server(port: int) -> HTTPServer:
    """Serve health checks from a daemon thread so it never blocks shutdown."""
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("health server listening on :%d", port)
    return server


def ping(url: str | None) -> bool:
    """Request our own public URL to stay awake. Never raises — this is best effort."""
    if not url:
        return False
    try:
        httpx.get(url, timeout=10)
        return True
    except Exception as exc:  # noqa: BLE001 - keep-alive must never crash the bot
        log.debug("keep-alive ping failed (non-critical): %s", exc)
        return False
