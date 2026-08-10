"""Schema invariant: memory_links must have no FK on ``entity_id``.

Locks in migration ``c4f7a2e9b6d1``. ``fk_memory_links_entity_id_entities``
was an ``ON DELETE CASCADE`` FK from ``memory_links.entity_id`` to
``entities.id``, left behind after ``e9b2c7d1f3a4`` deleted the last
``link_type = 'entity'`` row and ``e1b2c3d4f5a6`` dropped the last index
leading on ``entity_id``.

With no such index, PostgreSQL's referential-integrity trigger sequentially
scans all of ``memory_links`` once per deleted ``entities`` row — measured at
1.278 s per row on a ~2.7 GB / 26.7M-row table. That made
``prune_orphan_entities`` (graph maintenance Pass 2) exceed the 60 s asyncpg
``command_timeout`` on any bank with more than ~47 orphan entities, roll back,
retry nine times, and never make progress — which also starved
``prune_stale_cooccurrences`` in the same transaction.

The FK protected nothing (``entity_id`` is NULL in every row and no writer sets
it), so it is gone. If a future migration re-adds any FK on this column, this
test fails — re-introducing the outage would otherwise be invisible until a
bank accumulated orphans in production.

The column itself deliberately stays: it is a key column of
``idx_memory_links_unique``, which arbitrates retain's ``ON CONFLICT``. That is
asserted here too, so a future "cleanup" that drops the column has to come here
and think about the ``ON CONFLICT`` target first.
"""

import asyncpg
import pytest


@pytest.mark.asyncio
async def test_memory_links_entity_id_has_no_foreign_key(pg0_db_url):
    """No FK constraint may reference memory_links.entity_id."""
    conn = await asyncpg.connect(pg0_db_url)
    try:
        rows = await conn.fetch(
            """
            SELECT c.conname
            FROM pg_constraint c
            JOIN pg_attribute a
              ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
            WHERE c.conrelid = 'public.memory_links'::regclass
              AND c.contype = 'f'
              AND a.attname = 'entity_id'
            """
        )
    finally:
        await conn.close()

    assert rows == [], (
        "memory_links.entity_id must carry no FOREIGN KEY: "
        f"found {[r['conname'] for r in rows]}. Migration c4f7a2e9b6d1 dropped "
        "fk_memory_links_entity_id_entities because its ON DELETE CASCADE, "
        "unbacked by any index leading on entity_id, seq-scanned the whole "
        "table per deleted entity and made prune_orphan_entities time out "
        "permanently. Do not re-add an FK here without first adding an index "
        "on memory_links(entity_id)."
    )


@pytest.mark.asyncio
async def test_memory_links_entity_id_column_still_exists(pg0_db_url):
    """The column stays — idx_memory_links_unique keys on COALESCE(entity_id, ...)."""
    conn = await asyncpg.connect(pg0_db_url)
    try:
        col = await conn.fetchval(
            """
            SELECT a.attname
            FROM pg_attribute a
            WHERE a.attrelid = 'public.memory_links'::regclass
              AND a.attname = 'entity_id'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """
        )
        unique_index = await conn.fetchval(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'memory_links'
              AND indexname = 'idx_memory_links_unique'
            """
        )
    finally:
        await conn.close()

    assert col == "entity_id", (
        "memory_links.entity_id was dropped. That is a bigger change than it "
        "looks: it drops idx_memory_links_unique, which is the arbiter of the "
        "ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, "
        "...)) in PostgresOps.bulk_insert_links. Dropping the column requires "
        "rebuilding that unique index CONCURRENTLY and changing the ON CONFLICT "
        "target in ops.py, ops_postgresql.py and ops_oracle.py in the same "
        "deploy, or every link insert on an old pod fails."
    )
    assert unique_index is not None and "entity_id" in unique_index, (
        "idx_memory_links_unique must still key on entity_id via COALESCE — "
        f"got {unique_index!r}. See PostgresOps.bulk_insert_links."
    )
