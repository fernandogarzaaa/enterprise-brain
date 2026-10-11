"""Password hashing, session tokens (JWT), API keys and login rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

ALGORITHM = "HS256"
API_KEY_PREFIX = "eb"
_BCRYPT_MAX_BYTES = 72


# ---------------------------------------------------------------- passwords


def _password_bytes(password: str) -> bytes:
    raw = password.encode("utf-8")
    if len(raw) > _BCRYPT_MAX_BYTES:
        # bcrypt ignores bytes past 72; pre-hash so long passwords keep full entropy.
        raw = hashlib.sha256(raw).hexdigest().encode("ascii")
    return raw


def hash_password(password: str, rounds: int = 12) -> str:
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt(rounds=rounds)).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_password_bytes(password), hashed.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return False


def validate_password(password: str, min_length: int) -> str | None:
    """Return an error message when the password is too weak, else None."""
    if len(password) < min_length:
        return f"Password must be at least {min_length} characters"
    if len(password) > 256:
        return "Password must be at most 256 characters"
    if password.isalpha() or password.isdigit():
        return "Password must mix letters with digits or symbols"
    return None


def generate_temporary_password() -> str:
    # 16 url-safe chars plus a digit and a letter, so it always passes validate_password.
    return secrets.token_urlsafe(12) + str(secrets.randbelow(10)) + "a"


# ---------------------------------------------------------------- session tokens


@dataclass(frozen=True)
class TokenClaims:
    user_id: int
    token_version: int


def create_access_token(user_id: int, token_version: int, secret: str, expires_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "ver": token_version,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> TokenClaims | None:
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM], options={"require": ["sub", "exp"]})
        return TokenClaims(user_id=int(payload["sub"]), token_version=int(payload.get("ver", 0)))
    except (jwt.PyJWTError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------- API keys


@dataclass(frozen=True)
class NewApiKey:
    plaintext: str
    prefix: str
    key_hash: str


def hash_api_key(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def generate_api_key() -> NewApiKey:
    """Return a new key ``eb_<prefix>_<secret>``; store only prefix + hash."""
    prefix = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32)
    plaintext = f"{API_KEY_PREFIX}_{prefix}_{secret}"
    return NewApiKey(plaintext=plaintext, prefix=prefix, key_hash=hash_api_key(plaintext))


def parse_api_key(value: str) -> str | None:
    """Return the key's prefix when ``value`` looks like an API key, else None."""
    parts = value.split("_", 2)
    if len(parts) != 3 or parts[0] != API_KEY_PREFIX or len(parts[1]) != 12 or not parts[2]:
        return None
    return parts[1]


def api_key_matches(value: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(value), stored_hash)


# ---------------------------------------------------------------- rate limiting


class RateLimiter:
    """Sliding-window limiter keyed by an arbitrary string (in-process memory).

    With several worker processes each keeps its own window, so the effective limit
    is ``limit x workers``; put a reverse-proxy limit in front for strict guarantees.
    """

    def __init__(self, limit: int, window_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        return hits

    def is_limited(self, key: str) -> bool:
        with self._lock:
            return len(self._prune(key, self._clock())) >= self.limit

    def hit(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            self._prune(key, now).append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)

    def retry_after(self, key: str) -> int:
        with self._lock:
            now = self._clock()
            hits = self._prune(key, now)
            if len(hits) < self.limit:
                return 0
            return max(1, int(self.window - (now - hits[0])) + 1)
