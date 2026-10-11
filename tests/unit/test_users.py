from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from enterprise_brain import models, users
from enterprise_brain.config import Settings
from enterprise_brain.deps import AppContext
from enterprise_brain.security import verify_password
from enterprise_brain.users import UserError
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD


def admin(db: Session) -> models.User:
    u = users.get_by_email(db, ADMIN_EMAIL)
    assert u is not None
    return u


@pytest.mark.parametrize("bad", ["", "no-at", "a@b", "a b@c.d", "x" * 250 + "@ex.com"])
def test_normalise_email_rejects(bad: str) -> None:
    with pytest.raises(UserError, match="valid email"):
        users.normalise_email(bad)


def test_normalise_email() -> None:
    assert users.normalise_email("  Alice@Example.COM ") == "alice@example.com"


def test_bootstrap_admin_created(ctx: AppContext, db: Session) -> None:
    a = admin(db)
    assert a.role == "admin" and a.is_active and a.name == "Admin"
    assert verify_password(ADMIN_PASSWORD, a.hashed_password)
    assert users.ensure_admin(db, ctx.settings) is None  # idempotent


def test_bootstrap_skipped_without_credentials(settings_factory: object, db: Session) -> None:
    s: Settings = settings_factory(ADMIN_EMAIL="", ADMIN_PASSWORD="")  # type: ignore[operator]
    db.query(models.User).delete()
    db.commit()
    assert users.ensure_admin(db, s) is None
    assert db.query(models.User).count() == 0


def test_create_user_rules(ctx: AppContext, db: Session) -> None:
    s = ctx.settings
    u = users.create_user(db, s, "Bob@Example.com", "bob-password-1")
    assert (u.email, u.name, u.role, u.token_version, u.is_active) == ("bob@example.com", "bob", "member", 0, True)
    named = users.create_user(db, s, "carol@example.com", "carol-password-1", name="  Carol C  ", role="admin")
    assert (named.name, named.role) == ("Carol C", "admin")
    with pytest.raises(UserError, match="already a member") as dup:
        users.create_user(db, s, "BOB@example.com", "another-pass-1")
    assert dup.value.status == 409
    with pytest.raises(UserError, match="Role must be one of: admin, member"):
        users.create_user(db, s, "x@example.com", "x-password-12", role="owner")
    with pytest.raises(UserError, match="at least 10"):
        users.create_user(db, s, "y@example.com", "short1")
    weak = users.create_user(db, s, "z@example.com", "short", enforce_policy=False)
    assert verify_password("short", weak.hashed_password)


def test_user_error_default_status() -> None:
    assert UserError("x").status == 400


def test_authenticate(ctx: AppContext, db: Session) -> None:
    a = users.authenticate(db, " ADMIN@example.com ", ADMIN_PASSWORD)
    assert a is not None and a.last_login_at is not None
    assert users.authenticate(db, ADMIN_EMAIL, "wrong") is None
    assert users.authenticate(db, "ghost@example.com", ADMIN_PASSWORD) is None
    a.is_active = False
    db.commit()
    assert users.authenticate(db, ADMIN_EMAIL, ADMIN_PASSWORD) is None


def test_change_and_reset_password(ctx: AppContext, db: Session) -> None:
    a = admin(db)
    with pytest.raises(UserError, match="incorrect") as e:
        users.change_password(db, ctx.settings, a, "nope", "new-password-1")
    assert e.value.status == 403
    with pytest.raises(UserError, match="mix letters"):
        users.change_password(db, ctx.settings, a, ADMIN_PASSWORD, "abcdefghijkl")
    assert a.token_version == 0
    users.change_password(db, ctx.settings, a, ADMIN_PASSWORD, "new-password-1")
    assert a.token_version == 1
    assert verify_password("new-password-1", a.hashed_password)
    temp = users.reset_password(db, ctx.settings, a)
    assert a.token_version == 2
    assert verify_password(temp, a.hashed_password)


def test_update_member_guards(ctx: AppContext, db: Session) -> None:
    a = admin(db)
    m = users.create_user(db, ctx.settings, "m@example.com", "member-pass-1")
    with pytest.raises(UserError, match="cannot demote or deactivate yourself"):
        users.update_member(db, a, a, role="member")
    with pytest.raises(UserError, match="cannot demote or deactivate yourself"):
        users.update_member(db, a, a, is_active=False)
    with pytest.raises(UserError, match="Role must be"):
        users.update_member(db, a, m, role="root")
    users.update_member(db, a, a, name="  Boss ", role="admin", is_active=True)
    assert a.name == "Boss" and a.token_version == 0
    users.update_member(db, a, a, name="   ")
    assert a.name == "Boss"
    users.update_member(db, a, m, role="admin")
    assert m.role == "admin"
    # m demotes a: allowed while another active admin (m) remains
    users.update_member(db, m, a, role="member")
    assert a.role == "member"
    with pytest.raises(UserError, match="At least one active admin"):
        users.update_member(db, a, m, role="member")  # actor a is now a member, but m is last admin
    users.update_member(db, m, a, is_active=False)
    assert a.is_active is False and a.token_version == 1
    users.update_member(db, m, a, is_active=False)  # already inactive: no extra bump
    assert a.token_version == 1
    users.update_member(db, m, a, is_active=True)
    reactivated = admin(db)
    assert reactivated.is_active is True
    assert reactivated.token_version == 1


def test_last_admin_cannot_be_deactivated_by_other(ctx: AppContext, db: Session) -> None:
    a = admin(db)
    m = users.create_user(db, ctx.settings, "m2@example.com", "member-pass-1")
    with pytest.raises(UserError, match="At least one active admin"):
        users.update_member(db, m, a, is_active=False)
    with pytest.raises(UserError, match="At least one active admin"):
        users.update_member(db, m, a, role="member")
    assert users.active_admin_count(db) == 1


def test_inactive_admin_demotion_allowed(ctx: AppContext, db: Session) -> None:
    a = admin(db)
    other = users.create_user(db, ctx.settings, "o@example.com", "other-pass-1", role="admin")
    users.update_member(db, a, other, is_active=False)
    users.update_member(db, a, other, role="member")  # inactive admin can be demoted
    assert other.role == "member"


def test_delete_member(ctx: AppContext, db: Session) -> None:
    a = admin(db)
    m = users.create_user(db, ctx.settings, "d@example.com", "delete-pass-1")
    with pytest.raises(UserError, match="cannot delete yourself"):
        users.delete_member(db, a, a)
    with pytest.raises(UserError, match="At least one active admin"):
        users.delete_member(db, m, a)
    users.create_api_key(db, m, "k")
    users.delete_member(db, a, m)
    assert users.get_by_email(db, "d@example.com") is None
    assert db.query(models.ApiKey).count() == 0
    second = users.create_user(db, ctx.settings, "a2@example.com", "admin2-pass-1", role="admin")
    users.delete_member(db, second, a)  # another admin remains
    assert users.get_by_email(db, ADMIN_EMAIL) is None


def test_api_keys(ctx: AppContext, db: Session) -> None:
    a = admin(db)
    with pytest.raises(UserError, match="Key name is required"):
        users.create_api_key(db, a, "   ")
    created = users.create_api_key(db, a, "  CI   bot ")
    assert created.key.name == "CI bot"
    assert users.user_for_api_key(db, created.plaintext) is a
    assert created.key.last_used_at is not None
    assert users.user_for_api_key(db, created.plaintext + "x") is None
    assert users.user_for_api_key(db, "eb_000000000000_nope") is None
    assert users.user_for_api_key(db, "not a key") is None
    a.is_active = False
    db.commit()
    assert users.user_for_api_key(db, created.plaintext) is None
    a.is_active = True
    db.commit()
    users.revoke_api_key(db, a, created.key.id)
    revoked_at = created.key.revoked_at
    assert revoked_at is not None
    users.revoke_api_key(db, a, created.key.id)  # idempotent
    assert created.key.revoked_at == revoked_at
    assert users.user_for_api_key(db, created.plaintext) is None
    with pytest.raises(UserError, match="not found") as e:
        users.revoke_api_key(db, a, 9999)
    assert e.value.status == 404


def test_api_key_limit(ctx: AppContext, db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(users, "MAX_API_KEYS_PER_USER", 2)
    a = admin(db)
    k1 = users.create_api_key(db, a, "one")
    users.create_api_key(db, a, "two")
    with pytest.raises(UserError, match="At most 2"):
        users.create_api_key(db, a, "three")
    users.revoke_api_key(db, a, k1.key.id)
    assert users.create_api_key(db, a, "three").key.name == "three"


def test_revoke_other_users_key_not_found(ctx: AppContext, db: Session) -> None:
    a = admin(db)
    m = users.create_user(db, ctx.settings, "k@example.com", "keyuser-pass-1")
    key = users.create_api_key(db, m, "mine")
    with pytest.raises(UserError, match="not found"):
        users.revoke_api_key(db, a, key.key.id)
