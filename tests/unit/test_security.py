from __future__ import annotations

import time

import jwt
import pytest

from enterprise_brain.security import (
    ALGORITHM,
    RateLimiter,
    api_key_matches,
    create_access_token,
    decode_access_token,
    generate_api_key,
    generate_temporary_password,
    hash_api_key,
    hash_password,
    parse_api_key,
    validate_password,
    verify_password,
)

SECRET = "s" * 64


def test_hash_and_verify() -> None:
    h = hash_password("correct horse 1", rounds=4)
    assert h.startswith("$2b$04$")
    assert verify_password("correct horse 1", h)
    assert not verify_password("correct horse 2", h)
    assert hash_password("correct horse 1", rounds=4) != h  # salted


def test_default_rounds() -> None:
    assert hash_password("x1").startswith("$2b$12$")


def test_long_password_keeps_entropy() -> None:
    base = "a" * 80
    h = hash_password(base + "1", rounds=4)
    assert verify_password(base + "1", h)
    assert not verify_password(base + "2", h)  # bytes past 72 still matter
    exact = "b" * 72
    h72 = hash_password(exact, rounds=4)
    assert verify_password(exact, h72)
    assert not verify_password(exact + "c", h72)


def test_verify_beta_bcrypt_hash() -> None:
    import bcrypt

    legacy = bcrypt.hashpw(b"ChangeMe123!", bcrypt.gensalt(4)).decode()
    assert verify_password("ChangeMe123!", legacy)


@pytest.mark.parametrize("bad", ["", "not-a-hash", "$2b$04$short"])
def test_verify_garbage_hash(bad: str) -> None:
    assert verify_password("x", bad) is False


def test_validate_password() -> None:
    assert validate_password("short1", 10) == "Password must be at least 10 characters"
    assert validate_password("abcdefghij", 10) == "Password must mix letters with digits or symbols"
    assert validate_password("1234567890", 10) == "Password must mix letters with digits or symbols"
    assert validate_password("abcdefghi1", 10) is None
    assert validate_password("abcdefghi!", 10) is None
    assert validate_password("a1" * 128, 10) is None
    assert validate_password("a1" * 128 + "x", 10) == "Password must be at most 256 characters"


def test_temporary_password_is_valid() -> None:
    for _ in range(20):
        pw = generate_temporary_password()
        assert len(pw) == 18
        assert validate_password(pw, 10) is None
    assert generate_temporary_password() != generate_temporary_password()


def test_token_roundtrip() -> None:
    tok = create_access_token(7, 3, SECRET, 10)
    claims = decode_access_token(tok, SECRET)
    assert claims is not None
    assert (claims.user_id, claims.token_version) == (7, 3)
    payload = jwt.decode(tok, SECRET, algorithms=[ALGORITHM])
    assert payload["exp"] - payload["iat"] == 600
    assert payload["sub"] == "7"
    assert payload["ver"] == 3


def test_token_rejects_wrong_secret_expired_and_garbage() -> None:
    tok = create_access_token(1, 0, SECRET, 10)
    assert decode_access_token(tok, "other" * 10) is None
    assert decode_access_token("garbage", SECRET) is None
    expired = jwt.encode({"sub": "1", "exp": int(time.time()) - 5}, SECRET, algorithm=ALGORITHM)
    assert decode_access_token(expired, SECRET) is None
    no_exp = jwt.encode({"sub": "1"}, SECRET, algorithm=ALGORITHM)
    assert decode_access_token(no_exp, SECRET) is None
    no_sub = jwt.encode({"exp": int(time.time()) + 60}, SECRET, algorithm=ALGORITHM)
    assert decode_access_token(no_sub, SECRET) is None
    bad_sub = jwt.encode({"sub": "abc", "exp": int(time.time()) + 60}, SECRET, algorithm=ALGORITHM)
    assert decode_access_token(bad_sub, SECRET) is None
    other_alg = jwt.encode({"sub": "1", "exp": int(time.time()) + 60}, SECRET, algorithm="HS512")
    assert decode_access_token(other_alg, SECRET) is None


def test_token_without_version_defaults_zero() -> None:
    tok = jwt.encode({"sub": "5", "exp": int(time.time()) + 60}, SECRET, algorithm=ALGORITHM)
    claims = decode_access_token(tok, SECRET)
    assert claims is not None and claims.token_version == 0


def test_api_key_lifecycle() -> None:
    k = generate_api_key()
    assert k.plaintext.startswith(f"eb_{k.prefix}_")
    assert len(k.prefix) == 12
    assert len(k.plaintext) > 50
    assert parse_api_key(k.plaintext) == k.prefix
    assert k.key_hash == hash_api_key(k.plaintext)
    assert len(k.key_hash) == 64
    assert api_key_matches(k.plaintext, k.key_hash)
    assert not api_key_matches(k.plaintext + "x", k.key_hash)
    assert generate_api_key().prefix != k.prefix


def test_hash_api_key_value() -> None:
    assert hash_api_key("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize(
    "value",
    ["", "eb", "eb_abc_def", "xx_123456789012_secret", "eb_123456789012_", "eb_1234567890123_s", "bearer"],
)
def test_parse_api_key_rejects(value: str) -> None:
    assert parse_api_key(value) is None


def test_parse_api_key_secret_may_contain_underscores() -> None:
    assert parse_api_key("eb_123456789012_a_b_c") == "123456789012"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_window() -> None:
    clock = FakeClock()
    rl = RateLimiter(limit=3, window_seconds=60, clock=clock)
    for _ in range(2):
        rl.hit("k")
    assert not rl.is_limited("k")
    assert rl.retry_after("k") == 0
    rl.hit("k")
    assert rl.is_limited("k")
    assert not rl.is_limited("other")
    clock.now += 30
    assert rl.retry_after("k") == 31
    clock.now += 29.5
    assert rl.is_limited("k")
    clock.now += 0.5  # exactly window seconds after the first hit
    assert not rl.is_limited("k")


def test_rate_limiter_reset_and_retry_floor() -> None:
    clock = FakeClock()
    rl = RateLimiter(limit=1, window_seconds=10, clock=clock)
    rl.hit("k")
    clock.now += 9.9
    assert rl.retry_after("k") == 1
    rl.reset("k")
    assert not rl.is_limited("k")
    rl.reset("never-seen")  # no error


def test_rate_limiter_default_clock() -> None:
    rl = RateLimiter(limit=1, window_seconds=100)
    rl.hit("k")
    assert rl.is_limited("k")
    assert 1 <= rl.retry_after("k") <= 101
