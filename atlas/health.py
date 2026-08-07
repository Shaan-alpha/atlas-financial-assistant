"""Keep-alive for free hosting tiers.

Free web services sleep after ~15 minutes without HTTP traffic, and a sleeping
bot means a judge messages Atlas and waits ~50 seconds for a cold start — which
reads as broken. Two pieces solve it:

  1. A tiny HTTP server so the platform sees a healthy web service at all.
  2. A periodic self-request so the service never goes idle in the first place.
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

log = logging.getLogger(__name__)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"atlas ok")

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
