import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    gemini_api_key: str
    groq_api_key: str
    database_url: str
    log_level: str
    port: int
    public_url: str | None
    # Optional market-data providers. Each is tried in turn; absent keys are
    # skipped rather than erroring, so the bot runs with none of them set.
    finnhub_api_key: str | None
    fmp_api_key: str | None
    alphavantage_api_key: str | None


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def _normalize_database_url(raw: str) -> str:
    """Accept the postgres:// URL hosts hand out; SQLAlchemy needs a driver name."""
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql://", 1)
    if raw.startswith("postgresql://"):
        raw = raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw


def _normalize_public_url(raw: str | None) -> str | None:
    """Hosts display the bare hostname, and httpx refuses a URL with no scheme —
    which turns the keep-alive into a silent no-op."""
    if not raw:
        return None
    return raw if "://" in raw else f"https://{raw}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        telegram_token=_required("TELEGRAM_TOKEN"),
        gemini_api_key=_required("GEMINI_API_KEY"),
        groq_api_key=_required("GROQ_API_KEY"),
        database_url=_normalize_database_url(
            os.environ.get("DATABASE_URL", "sqlite:///atlas.db")
        ),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        port=int(os.environ.get("PORT", "8080")),
        public_url=_normalize_public_url(os.environ.get("PUBLIC_URL")),
        finnhub_api_key=os.environ.get("FINNHUB_API_KEY") or None,
        fmp_api_key=os.environ.get("FMP_API_KEY") or None,
        alphavantage_api_key=os.environ.get("ALPHAVANTAGE_API_KEY") or None,
    )
