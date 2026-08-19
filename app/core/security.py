"""Security & Auth Helpers Module.

Handles JWT bearer token verification.
"""

import logging
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from app.core.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> str:
    """Extracts and cryptographically verifies user identity from the Authorization header.

    No token present -> "system_anonymous_user" (routes are not yet gated on auth; see
    JWT_SECRET_KEY note below). A token that IS present but fails signature/expiry
    verification is rejected outright -- unlike the prior placeholder, which returned
    "user_authenticated" for any non-empty string with no verification at all, meaning a
    forged or expired token was silently accepted as valid.
    """
    if not credentials or not credentials.credentials:
        return "system_anonymous_user"

    if not settings.JWT_SECRET_KEY:
        # No signing secret configured: we cannot verify anything cryptographically, so a
        # provided token must not be trusted just because it exists (that was the prior
        # bug). Fail closed rather than accept it as if it were checked.
        logger.error("Bearer token presented but JWT_SECRET_KEY is not configured; rejecting")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "AUTH_NOT_CONFIGURED", "message": "Token verification is not configured"},
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "INVALID_TOKEN", "message": f"Invalid or expired token: {exc}"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "INVALID_TOKEN", "message": "Token missing 'sub' claim"},
        )
    return str(user_id)
