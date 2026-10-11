"""User and API-key management."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.config import Settings
from enterprise_brain.models import utcnow
from enterprise_brain.security import (
    api_key_matches,
    generate_api_key,
    generate_temporary_password,
    hash_password,
    parse_api_key,
    validate_password,
    verify_password,
)

log = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_API_KEYS_PER_USER = 20


class UserError(ValueError):
    """Invalid user operation. ``status`` is the HTTP status it maps to."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def normalise_email(email: str) -> str:
    email = email.strip().lower()
    if len(email) > 255 or not _EMAIL_RE.match(email):
        raise UserError("A valid email address is required")
    return email


def get_by_email(db: Session, email: str) -> models.User | None:
    return db.query(models.User).filter(models.User.email == email.strip().lower()).one_or_none()


def create_user(
    db: Session,
    settings: Settings,
    email: str,
    password: str,
    name: str = "",
    role: str = "member",
    *,
    enforce_policy: bool = True,
) -> models.User:
    email = normalise_email(email)
    if role not in models.ROLES:
        raise UserError(f"Role must be one of: {', '.join(models.ROLES)}")
    if enforce_policy:
        problem = validate_password(password, settings.MIN_PASSWORD_LENGTH)
        if problem:
            raise UserError(problem)
    if get_by_email(db, email):
        raise UserError("That email is already a member", status=409)
    user = models.User(
        email=email,
        name=(name.strip() or email.split("@")[0])[:255],
        hashed_password=hash_password(password, settings.BCRYPT_ROUNDS),
        role=role,
        is_active=True,
        token_version=0,
    )
    db.add(user)
    db.commit()
    return user


def authenticate(db: Session, email: str, password: str) -> models.User | None:
    user = get_by_email(db, email)
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    user.last_login_at = utcnow()
    db.commit()
    return user


def change_password(db: Session, settings: Settings, user: models.User, current: str, new: str) -> None:
    if not verify_password(current, user.hashed_password):
        raise UserError("Current password is incorrect", status=403)
    problem = validate_password(new, settings.MIN_PASSWORD_LENGTH)
    if problem:
        raise UserError(problem)
    user.hashed_password = hash_password(new, settings.BCRYPT_ROUNDS)
    user.token_version += 1  # sign out every other session
    db.commit()


def reset_password(db: Session, settings: Settings, user: models.User) -> str:
    temp = generate_temporary_password()
    user.hashed_password = hash_password(temp, settings.BCRYPT_ROUNDS)
    user.token_version += 1
    db.commit()
    return temp


def active_admin_count(db: Session) -> int:
    return int(
        db.query(func.count(models.User.id))
        .filter(models.User.role == "admin", models.User.is_active.is_(True))
        .scalar()
        or 0
    )


def update_member(
    db: Session,
    actor: models.User,
    member: models.User,
    *,
    role: str | None = None,
    is_active: bool | None = None,
    name: str | None = None,
) -> models.User:
    if role is not None and role not in models.ROLES:
        raise UserError(f"Role must be one of: {', '.join(models.ROLES)}")
    demoting = role is not None and role != "admin" and member.role == "admin"
    deactivating = is_active is False and member.is_active
    if member.id == actor.id and (demoting or deactivating):
        raise UserError("You cannot demote or deactivate yourself")
    if member.is_active and (demoting or (deactivating and member.role == "admin")) and active_admin_count(db) <= 1:
        raise UserError("At least one active admin is required")
    if role is not None:
        member.role = role
    if is_active is not None:
        if deactivating:
            member.token_version += 1
        member.is_active = is_active
    if name is not None and name.strip():
        member.name = name.strip()[:255]
    db.commit()
    return member


def delete_member(db: Session, actor: models.User, member: models.User) -> None:
    if member.id == actor.id:
        raise UserError("You cannot delete yourself")
    if member.role == "admin" and member.is_active and active_admin_count(db) <= 1:
        raise UserError("At least one active admin is required")
    db.delete(member)
    db.commit()


def ensure_admin(db: Session, settings: Settings) -> models.User | None:
    """Create the bootstrap admin from ADMIN_EMAIL/ADMIN_PASSWORD when no users exist."""
    if db.query(models.User.id).first() is not None:
        return None
    password = settings.ADMIN_PASSWORD.get_secret_value()
    if not settings.ADMIN_EMAIL or not password:
        log.warning(
            "no users exist and ADMIN_EMAIL/ADMIN_PASSWORD are not set; "
            "create one with `enterprise-brain create-admin`"
        )
        return None
    user = create_user(
        db,
        settings,
        settings.ADMIN_EMAIL,
        password,
        name="Admin",
        role="admin",
        enforce_policy=False,  # production strength is enforced by Settings validation
    )
    log.info("bootstrap admin created", extra={"email": user.email})
    return user


# ---------------------------------------------------------------- API keys


@dataclass(frozen=True)
class CreatedKey:
    key: models.ApiKey
    plaintext: str


def create_api_key(db: Session, user: models.User, name: str) -> CreatedKey:
    name = " ".join(name.split())[:100]
    if not name:
        raise UserError("Key name is required")
    active = (
        db.query(func.count(models.ApiKey.id))
        .filter(models.ApiKey.user_id == user.id, models.ApiKey.revoked_at.is_(None))
        .scalar()
        or 0
    )
    if active >= MAX_API_KEYS_PER_USER:
        raise UserError(f"At most {MAX_API_KEYS_PER_USER} active API keys per user")
    new = generate_api_key()
    key = models.ApiKey(user_id=user.id, name=name, prefix=new.prefix, key_hash=new.key_hash)
    db.add(key)
    db.commit()
    return CreatedKey(key=key, plaintext=new.plaintext)


def user_for_api_key(db: Session, value: str) -> models.User | None:
    prefix = parse_api_key(value)
    if prefix is None:
        return None
    key = db.query(models.ApiKey).filter(models.ApiKey.prefix == prefix).one_or_none()
    if key is None or key.revoked_at is not None or not api_key_matches(value, key.key_hash):
        return None
    if not key.user.is_active:
        return None
    key.last_used_at = utcnow()
    db.commit()
    return key.user


def revoke_api_key(db: Session, user: models.User, key_id: int) -> None:
    key = (
        db.query(models.ApiKey)
        .filter(models.ApiKey.id == key_id, models.ApiKey.user_id == user.id)
        .one_or_none()
    )
    if key is None:
        raise UserError("API key not found", status=404)
    if key.revoked_at is None:
        key.revoked_at = utcnow()
        db.commit()
