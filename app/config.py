"""Application configuration via environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        env_prefix="DOCSHIELD_",
        extra="ignore",
    )

    env: str = "development"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    data_dir: Path = ROOT_DIR / "data"
    upload_dir: Path = ROOT_DIR / "data" / "uploads"
    temp_dir: Path = ROOT_DIR / "data" / "temp"
    db_url: str = f"sqlite:///{(ROOT_DIR / 'data' / 'docshield.db').as_posix()}"

    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    allowed_extensions: str = ".png,.jpg,.jpeg,.webp,.pdf"

    # Secure retention: prefer not keeping raw identity images permanently
    allow_pdf: bool = True
    retain_uploads: bool = False
    upload_ttl_seconds: int = 3600  # purge aged files when retain_uploads=true
    rate_limit_per_minute: int = 30

    demo_mode: bool = True
    api_key: str = ""

    official_verify_url: str = ""
    reverse_search_url: str = ""

    @property
    def allowed_extension_list(self) -> List[str]:
        exts = [e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()]
        if not self.allow_pdf:
            exts = [e for e in exts if e != ".pdf"]
        return exts

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.temp_dir, ROOT_DIR / "data" / "reports"):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
