"""Optional API-key dependency for mutating routes."""

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def require_api_key_if_configured(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
