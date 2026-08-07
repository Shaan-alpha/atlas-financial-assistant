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


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        telegram_token=_required("TELEGRAM_TOKEN"),
        gemini_api_key=_required("GEMINI_API_KEY"),
        groq_api_key=_required("GROQ_API_KEY"),
        database_url=os.environ.get("DATABASE_URL", "sqlite:///atlas.db"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
