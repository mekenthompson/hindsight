"""Drop the vestigial ``memory_links.entity_id`` → ``entities.id`` FK.

``fk_memory_links_entity_id_entities`` was installed by the initial schema
migration (``5a366d414dce``) with ``ON DELETE CASCADE``, back when entity edges
were materialised as ``memory_links`` rows with ``link_type = 'entity'``. That
representation is gone: ``e1b2c3d4f5a6`` dropped the entity index and
``e9b2c7d1f3a4`` deleted every ``link_type = 'entity'`` row, and no writer has
populated ``entity_id`` since — ``_bulk_insert_links`` passes ``None`` for every
row it inserts. On the bank this was diagnosed against,
``SELECT count(*) FROM memory_links WHERE entity_id IS NOT NULL`` returns 0 out
of 26,750,591 rows and ``pg_stats.null_frac`` for the column is 1.

The constraint protects nothing, but it is not free. ``memory_links`` has no
index leading on ``entity_id`` (``e1b2c3d4f5a6`` dropped the last one), so
PostgreSQL's referential-integrity trigger has to sequentially scan the whole
table once per deleted ``entities`` row. Measured on a fully-cached ~2.7 GB
``memory_links``: **1.278 s per parent row deleted**.

That makes ``prune_orphan_entities`` (graph maintenance Pass 2) unrunnable on
any bank with a real orphan backlog. The anti-join that finds the orphans is
cheap (~70 ms, indexed by ``idx_unit_entities_entity_unit``), but the DELETE it
drives pays the cascade scan per row: ~157 orphans ≈ 200 s against asyncpg's
60 s client-side ``command_timeout`` (``DEFAULT_DB_COMMAND_TIMEOUT``). The sweep
times out, rolls back, and retries — ``_SWEEP_MAX_RETRIES = 8`` means nine
attempts and ~560 s of wasted work per job. Because it rolls back it never makes
partial progress, so past roughly 47 orphans a bank can never recover on its
own, and ``prune_stale_cooccurrences`` (same transaction, runs second) stops
running entirely.

What this migration does:

* Drops ``fk_memory_links_entity_id_entities`` on PostgreSQL, idempotently
  (``DROP CONSTRAINT IF EXISTS``) — the constraint has already been dropped by
  hand on at least one production database as an emergency unblock, so the
  migration must reconcile a schema that is *ahead* of head rather than fail on
  it.

What this migration deliberately does NOT do: drop the ``entity_id`` column.
See the comment above ``_pg_schema_prefix`` below for why.

PostgreSQL-only, by the same reasoning as ``9f8e7d6c5b4a``: Oracle keeps
``idx_ml_entity`` on ``memory_links(entity_id)`` (``o1a2b3c4d5e6``), so its
``fk_ml_entity`` cascade is an index lookup, not a table scan — the pathology
being fixed here does not exist there, and the Oracle baseline still creates the
constraint as part of the table DDL.

Revision ID: c4f7a2e9b6d1
Revises: b3e8d1c6f4a9
Create Date: 2026-08-10
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c4f7a2e9b6d1"
down_revision: str | Sequence[str] | None = "b3e8d1c6f4a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FK_NAME = "fk_memory_links_entity_id_entities"

# Why the ``entity_id`` COLUMN stays, even though it is 100% NULL and nothing
# reads it:
#
# 1. It is a key column of ``idx_memory_links_unique``
#    ``(from_unit_id, to_unit_id, link_type, COALESCE(entity_id, nil_uuid))``,
#    which is the arbiter of the retain hot path's
#    ``ON CONFLICT ... DO NOTHING`` in ``PostgresOps.bulk_insert_links``.
#    Dropping the column drops that index, and rebuilding a unique index over
#    26.7M rows has to happen CONCURRENTLY or it takes an ACCESS EXCLUSIVE lock
#    on the busiest table in the schema for the duration of the build.
# 2. The ON CONFLICT target must then change in lockstep, in the same deploy, in
#    three places that are not deployed atomically with the migration:
#    ``ops.bulk_insert_links``' abstract signature (it takes ``nil_entity_uuid``),
#    ``ops_postgresql.bulk_insert_links``, and ``ops_oracle.bulk_insert_links``.
#    An old API pod talking to a new schema — the normal state of a rolling
#    deploy — would fail every link insert with "no unique or exclusion
#    constraint matching the ON CONFLICT specification", i.e. retain breaks.
# 3. Oracle's baseline (``o1a2b3c4d5e6``) declares the same column in the table
#    DDL and the same NVL-based unique index, so a column drop is a two-dialect
#    schema change, not a one-line PG cleanup.
#
# None of that buys anything the outage cares about: the cascade scan is caused
# by the *constraint*, not by the column. A 16-byte-nullable column that is NULL
# in every row costs 1 bit of null bitmap per row and nothing else. Removing the
# column is a worthwhile follow-up, but it is a separate, coordinated change
# with its own rollout story — not something to bundle into an outage fix.


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # IF EXISTS is load-bearing, not defensive boilerplate: production already
    # ran this ALTER by hand, so on that database the constraint is gone while
    # alembic_version still says b3e8d1c6f4a9. Without IF EXISTS the migration
    # would abort there and leave the version table permanently behind.
    #
    # This is a catalogue-only change: DROP CONSTRAINT takes a brief ACCESS
    # EXCLUSIVE lock on memory_links and does not touch the heap, so it is
    # instant regardless of table size.
    op.execute(f"ALTER TABLE {schema}memory_links DROP CONSTRAINT IF EXISTS {_FK_NAME}")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()

    # The restore is faithful *for the data that exists*: entity_id is NULL in
    # every row, and a NULL FK column is trivially satisfied, so re-adding the
    # constraint cannot fail on current data.
    #
    # It is NOT perfectly reversible in general. If a downgrade is ever run on a
    # database that somehow acquired non-NULL entity_id values pointing at
    # deleted entities while the constraint was absent, VALIDATE will fail and
    # the downgrade will abort — correctly. Nothing in the codebase can produce
    # such a row (every writer passes NULL), so this is a theoretical edge, but
    # the failure mode is "downgrade refuses" rather than "silent corruption",
    # which is the right way round.
    #
    # ADD ... NOT VALID then VALIDATE, rather than a plain ADD CONSTRAINT: a
    # validating ADD holds ACCESS EXCLUSIVE on memory_links (and SHARE ROW
    # EXCLUSIVE on entities) for a full seq scan of 26.7M rows. NOT VALID takes
    # the exclusive lock only for the catalogue write; VALIDATE then runs under
    # SHARE UPDATE EXCLUSIVE, which does not block reads or writes. The two must
    # be separate transactions, hence the autocommit block.
    #
    # Note the downgrade reinstates the cascade scan, and therefore reinstates
    # the prune_orphan_entities outage. That is what "restore the prior schema"
    # means here; it is intentional and it is why the upgrade direction is the
    # one you want.
    with op.get_context().autocommit_block():
        # Symmetric with the upgrade's IF EXISTS: a database that still has the
        # constraint (never upgraded, or re-added by hand) must not make the
        # downgrade blow up on "constraint already exists".
        op.execute(f"ALTER TABLE {schema}memory_links DROP CONSTRAINT IF EXISTS {_FK_NAME}")
        op.execute(
            f"""
            ALTER TABLE {schema}memory_links
                ADD CONSTRAINT {_FK_NAME}
                FOREIGN KEY (entity_id)
                REFERENCES {schema}entities (id)
                ON DELETE CASCADE
                NOT VALID
            """
        )
        op.execute(f"ALTER TABLE {schema}memory_links VALIDATE CONSTRAINT {_FK_NAME}")


def upgrade() -> None:
    # Oracle slot intentionally absent → no-op there. Oracle indexes
    # memory_links(entity_id) (idx_ml_entity), so its equivalent constraint
    # (fk_ml_entity) cascades via an index lookup and does not exhibit the
    # per-parent-row seq scan this migration exists to remove.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
