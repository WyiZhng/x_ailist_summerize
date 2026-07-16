from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.models import (
    IngestionRun,
    ListSyncState,
    XAuthor,
    XPost,
    XPostMetrics,
    XPostReference,
)
from app.storage import (
    FilePostStore,
    InMemoryPostStore,
    StorageConflictError,
    StorageCorruptionError,
    StorageOwnershipConflictError,
    StorageValidationError,
)


NOW = datetime(2026, 7, 16, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def tmp_path():
    """Use a workspace-owned temp directory on restricted Windows hosts."""

    path = Path.cwd() / "tests" / f".tmp-storage-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_post(
    post_id: str,
    *,
    list_id: str = "list-a",
    author_id: str = "author-1",
    author_name: str = "Alice",
) -> XPost:
    return XPost(
        id=post_id,
        text=f"post {post_id}",
        author_id=author_id,
        author=XAuthor(
            id=author_id,
            username=f"user_{author_id}",
            name=author_name,
            profile_image_url="https://example.test/avatar.png",
        ),
        created_at=NOW,
        language="en",
        conversation_id=f"conversation-{post_id}",
        source_list_id=list_id,
        metrics=XPostMetrics(
            likes=2,
            retweets=3,
            replies=4,
            quotes=5,
            bookmarks=6,
            impressions=7,
        ),
        references=[XPostReference("quoted", "older-post")],
        urls=["https://example.test/article"],
        media=[{"type": "photo", "url": "https://example.test/image.png"}],
        raw_payload={"id": post_id, "evidence": True},
        fetched_at=NOW,
    )


def make_state(
    list_id: str = "list-a",
    *,
    newest_post_id: str | None = "100",
    status: str = "success",
) -> ListSyncState:
    return ListSyncState(
        list_id=list_id,
        newest_post_id=newest_post_id,
        newest_post_created_at=NOW if newest_post_id else None,
        last_attempt_at=NOW,
        last_success_at=NOW if status == "success" else None,
        last_run_id="run-1",
        last_status=status,
        last_error=None,
    )


def make_run(run_id: str = "run-1", *, status: str = "running") -> IngestionRun:
    return IngestionRun(
        run_id=run_id,
        started_at=NOW,
        finished_at=None if status == "running" else NOW,
        status=status,
        requested_lists=["list-a", "list-b"],
        successful_lists=[] if status == "running" else ["list-a"],
        failed_lists=[] if status in {"running", "success"} else ["list-b"],
        fetched_count=2,
        new_post_count=1,
        duplicate_count=1,
        new_post_ids=["100"],
        errors=[],
    )


@pytest.fixture(params=["file", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "file":
        return FilePostStore(tmp_path / "data")
    return InMemoryPostStore()


def test_new_post_author_and_source_membership_are_saved(store) -> None:
    post = make_post("100")

    store.save_posts([post])

    assert store.has_post("100")
    assert store.has_membership("100", "list-a")
    assert not store.has_membership("100", "list-b")
    assert [item.id for item in store.read_posts()] == ["100"]
    assert [item.id for item in store.read_authors()] == ["author-1"]
    assert store.read_posts()[0].raw_payload == {"id": "100", "evidence": True}


def test_duplicate_post_is_global_but_memberships_are_per_list(store) -> None:
    first = make_post("100", list_id="list-a")
    second = make_post("100", list_id="list-b")

    store.save_posts([first, first, second])

    assert len(store.read_posts()) == 1
    assert store.has_post("100")
    assert store.has_membership("100", "list-a")
    assert store.has_membership("100", "list-b")
    assert [item.id for item in store.read_posts("list-a")] == ["100"]
    assert [item.id for item in store.read_posts("list-b")] == ["100"]
    assert len(store.read_memberships()) == 2


def test_global_post_does_not_imply_membership_in_another_list(store) -> None:
    store.save_posts([make_post("100", list_id="list-a")])

    assert store.has_post("100") is True
    assert store.has_membership("100", "list-b") is False


def test_author_is_upserted_once_by_id(store) -> None:
    store.save_posts([make_post("100", author_name="Old Name")])
    store.save_posts(
        [make_post("101", author_name="Current Name", author_id="author-1")]
    )

    authors = store.read_authors()
    assert len(authors) == 1
    assert authors[0].name == "Current Name"


def test_sync_states_are_independent_and_detached(store) -> None:
    state_a = make_state("list-a", newest_post_id="100")
    state_b = make_state("list-b", newest_post_id="200")

    store.save_sync_state(state_a)
    store.save_sync_state(state_b)
    state_a.newest_post_id = "mutated-after-save"

    assert store.get_sync_state("list-a").newest_post_id == "100"
    assert store.get_sync_state("list-b").newest_post_id == "200"
    assert store.get_sync_state("not-present") is None


def test_commit_list_batch_persists_entities_membership_and_state(store) -> None:
    state = make_state("list-a", newest_post_id="101")

    store.commit_list_batch([make_post("101", list_id="legacy-source")], "list-a", state)

    assert store.has_post("101")
    assert store.has_membership("101", "list-a")
    assert not store.has_membership("101", "legacy-source")
    assert store.get_sync_state("list-a").newest_post_id == "101"


def test_commit_list_batch_persists_report_ownership_before_checkpoint(store) -> None:
    run = make_run()
    store.create_run(run)

    inserted = store.commit_list_batch(
        [make_post("101")],
        "list-a",
        make_state("list-a", newest_post_id="101"),
        expected_newest_post_id=None,
        run=run,
    )

    stored_run = store.get_run(run.run_id)
    assert inserted == {"101"}
    assert stored_run.new_post_ids == ["100", "101"]
    assert stored_run.new_post_count == 2
    assert stored_run.successful_lists == ["list-a"]


def test_commit_rejects_mismatched_state_without_writes(store) -> None:
    with pytest.raises(StorageValidationError, match="must match"):
        store.commit_list_batch(
            [make_post("101")], "list-a", make_state("list-b", newest_post_id="101")
        )

    assert not store.has_post("101")
    assert store.get_sync_state("list-a") is None


def test_commit_rejects_a_stale_checkpoint_without_writes(store) -> None:
    store.save_sync_state(make_state("list-a", newest_post_id="100"))

    with pytest.raises(StorageConflictError, match="changed concurrently"):
        store.commit_list_batch(
            [make_post("101")],
            "list-a",
            make_state("list-a", newest_post_id="101"),
            expected_newest_post_id="99",
        )

    assert not store.has_post("101")
    assert store.get_sync_state("list-a").newest_post_id == "100"


def test_ingestion_run_create_update_and_read_are_separate_from_reports(store) -> None:
    running = make_run()
    store.create_run(running)
    running.status = "failed"  # The stored object must remain detached.

    assert store.get_run("run-1").status == "running"

    finished = make_run(status="partial_success")
    store.update_run(finished)

    assert store.get_run("run-1").status == "partial_success"
    assert [run.run_id for run in store.read_runs()] == ["run-1"]
    assert store.get_run("unknown") is None


def test_ingestion_run_report_status_update_uses_compare_and_swap(store) -> None:
    run = make_run()
    store.create_run(run)
    claimed = store.get_run(run.run_id)
    claimed.report_status = "generating"
    store.update_run(claimed, expected_report_status="not_started")

    stale = store.get_run(run.run_id)
    stale.report_status = "failed"
    with pytest.raises(StorageConflictError, match="report status changed"):
        store.update_run(stale, expected_report_status="not_started")

    assert store.get_run(run.run_id).report_status == "generating"


def test_ingestion_updates_preserve_a_concurrent_report_claim(store) -> None:
    run = make_run()
    store.create_run(run)
    stale = store.get_run(run.run_id)

    claimed = store.get_run(run.run_id)
    claimed.report_status = "generating"
    claimed.report_claim_id = "claim-new"
    claimed.report_claimed_at = NOW
    store.update_run(
        claimed,
        expected_report_status="not_started",
        expected_report_claim_id=None,
    )

    store.commit_list_batch(
        [make_post("100")],
        "list-a",
        make_state(),
        run=stale,
    )
    ingestion_update = store.get_run(run.run_id)
    ingestion_update.status = "success"
    ingestion_update.report_status = "not_started"
    ingestion_update.report_claim_id = None
    ingestion_update.report_claimed_at = None
    store.update_ingestion_run(ingestion_update)

    current = store.get_run(run.run_id)
    assert current.report_status == "generating"
    assert current.report_claim_id == "claim-new"
    assert current.report_claimed_at == NOW
    assert current.new_post_ids == ["100"]


def test_active_ingestion_owner_fences_stale_updates_and_commits(store) -> None:
    run = make_run()
    run.ingestion_owner_id = "owner-active"
    run.ingestion_heartbeat_at = NOW
    store.create_run(run)
    stale = store.get_run(run.run_id)

    reclaimed = store.get_run(run.run_id)
    reclaimed.ingestion_owner_id = "owner-reclaimed"
    store.update_ingestion_run(
        reclaimed,
        expected_ingestion_owner_id="owner-active",
    )

    with pytest.raises(StorageOwnershipConflictError, match="owner changed"):
        store.update_ingestion_run(
            stale,
            expected_ingestion_owner_id="owner-active",
        )

    with pytest.raises(StorageOwnershipConflictError, match="owner changed"):
        store.commit_list_batch(
            [make_post("101")],
            "list-a",
            make_state("list-a", newest_post_id="101"),
            expected_newest_post_id=None,
            run=stale,
        )

    assert store.get_run(run.run_id).ingestion_owner_id == "owner-reclaimed"
    assert not store.has_post("101")
    assert store.get_sync_state("list-a") is None


def test_ingestion_heartbeat_cas_prevents_recovering_a_renewed_owner(store) -> None:
    run = make_run()
    run.ingestion_owner_id = "owner-active"
    run.ingestion_heartbeat_at = NOW
    store.create_run(run)
    stale = store.get_run(run.run_id)

    renewed = store.get_run(run.run_id)
    renewed.ingestion_heartbeat_at = NOW + timedelta(minutes=1)
    store.update_ingestion_run(
        renewed,
        expected_ingestion_owner_id="owner-active",
        expected_ingestion_heartbeat_at=NOW,
    )

    stale.status = "failed"
    stale.ingestion_owner_id = None
    with pytest.raises(StorageOwnershipConflictError, match="heartbeat changed"):
        store.update_ingestion_run(
            stale,
            expected_ingestion_owner_id="owner-active",
            expected_ingestion_heartbeat_at=NOW,
        )

    current = store.get_run(run.run_id)
    assert current.status == "running"
    assert current.ingestion_owner_id == "owner-active"
    assert current.ingestion_heartbeat_at == NOW + timedelta(minutes=1)


def test_report_claim_compare_and_swap_checks_owner(store) -> None:
    run = make_run()
    run.report_status = "generating"
    run.report_claim_id = "claim-new"
    run.report_claimed_at = NOW
    store.create_run(run)
    stale = store.get_run(run.run_id)
    stale.report_status = "succeeded"
    stale.report_claim_id = None
    stale.report_claimed_at = None

    with pytest.raises(StorageConflictError, match="claim"):
        store.update_run(
            stale,
            expected_report_status="generating",
            expected_report_claim_id="claim-old",
        )

    assert store.get_run(run.run_id).report_claim_id == "claim-new"


def test_ingestion_errors_are_redacted_before_persistence(store) -> None:
    run = make_run(status="failed")
    run.errors = [
        {
            "message": "request failed with Bearer TEST_BEARER_TOKEN_1234567890",
            "api_key": "TEST_API_KEY_DO_NOT_USE",
            "nested": {"password": "TEST_PASSWORD_DO_NOT_USE"},
        }
    ]
    run.report_error = "sk-TESTONLY0123456789abcdefghijklmnop"

    store.create_run(run)
    stored = store.get_run(run.run_id)

    encoded = json.dumps(stored.to_dict())
    assert "TEST_BEARER_TOKEN" not in encoded
    assert "TEST_API_KEY" not in encoded
    assert "TEST_PASSWORD" not in encoded
    assert "sk-TESTONLY" not in encoded
    assert "[REDACTED]" in encoded


def test_ingestion_run_create_and_update_validate_lifecycle(store) -> None:
    with pytest.raises(StorageValidationError, match="does not exist"):
        store.update_run(make_run())

    store.create_run(make_run())
    with pytest.raises(StorageValidationError, match="already exists"):
        store.create_run(make_run())
    with pytest.raises(StorageValidationError, match="run_id"):
        store.get_run("../../escape")


def test_file_store_rejects_invalid_run_filename(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    invalid_path = store.runs_dir / "run_$.json"
    invalid_path.write_text(
        json.dumps({**make_run().to_dict(), "run_id": "$"}), encoding="utf-8"
    )

    with pytest.raises(StorageCorruptionError, match="run_id"):
        store.read_runs()


def test_export_data_is_json_compatible_and_detached(store) -> None:
    store.commit_list_batch([make_post("100")], "list-a", make_state())
    store.create_run(make_run())

    exported = store.export_data()
    encoded = json.dumps(exported, ensure_ascii=False, allow_nan=False)
    exported["posts"][0]["text"] = "changed"

    assert '"memberships"' in encoded
    assert store.read_posts()[0].text == "post 100"
    assert exported["sync_states"][0]["list_id"] == "list-a"


def test_file_store_uses_expected_separate_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = FilePostStore(data_dir)
    store.commit_list_batch([make_post("100")], "list-a", make_state())
    store.create_run(make_run())

    assert (data_dir / "posts" / "posts.jsonl").is_file()
    assert (data_dir / "authors" / "authors.jsonl").is_file()
    assert (data_dir / "list_memberships" / "post_lists.jsonl").is_file()
    assert (data_dir / "sync" / "list_sync_states.json").is_file()
    assert (data_dir / "runs" / "run_run-1.json").is_file()


def test_file_store_reports_damaged_final_jsonl_line(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_posts([make_post("100")])
    with store.posts_path.open("a", encoding="utf-8") as handle:
        handle.write('{"id":"broken"')

    with pytest.raises(StorageCorruptionError) as error:
        store.read_posts()

    message = str(error.value)
    assert str(store.posts_path) in message
    assert "final line" in message
    assert "line 2" in message


def test_file_store_recovers_an_interrupted_manifest_on_startup(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = FilePostStore(data_dir)
    store.save_posts([make_post("100")])
    original = store.posts_path.read_bytes()
    backup = store.posts_path.parent / ".posts.jsonl.simulated.backup.tmp"
    temporary = store.posts_path.parent / ".posts.jsonl.simulated.tmp"
    shutil.copyfile(store.posts_path, backup)
    temporary.write_text("interrupted replacement\n", encoding="utf-8")
    store._write_transaction_manifest([(store.posts_path, temporary, backup)])
    os.replace(temporary, store.posts_path)

    recovered = FilePostStore(data_dir)

    assert recovered.posts_path.read_bytes() == original
    assert recovered.read_posts()[0].id == "100"
    assert not recovered.transaction_manifest_path.exists()
    assert not backup.exists()


def test_existing_file_store_recovers_after_another_writer_crashes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = FilePostStore(data_dir)
    store.save_posts([make_post("100")])
    original = store.posts_path.read_bytes()
    backup = store.posts_path.parent / ".posts.jsonl.takeover.backup.tmp"
    temporary = store.posts_path.parent / ".posts.jsonl.takeover.tmp"
    shutil.copyfile(store.posts_path, backup)
    temporary.write_text("interrupted replacement\n", encoding="utf-8")
    store._write_transaction_manifest([(store.posts_path, temporary, backup)])
    os.replace(temporary, store.posts_path)

    # Reuse the already-created instance, mirroring a Web process that was
    # waiting on the crashed writer's process lock.
    assert [post.id for post in store.read_posts()] == ["100"]
    assert store.posts_path.read_bytes() == original
    assert not store.transaction_manifest_path.exists()
    assert not backup.exists()


def test_manifest_install_interruption_keeps_backups_for_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_posts([make_post("100")])
    original = store.posts_path.read_bytes()
    real_write_manifest = store._write_transaction_manifest

    def install_then_interrupt(staged):
        real_write_manifest(staged)
        raise KeyboardInterrupt("after manifest install")

    monkeypatch.setattr(store, "_write_transaction_manifest", install_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="manifest"):
        store.save_posts([make_post("101")])

    assert store.posts_path.read_bytes() == original
    assert [post.id for post in store.read_posts()] == ["100"]
    assert not store.transaction_manifest_path.exists()


def test_target_replace_interruption_rolls_back_the_completed_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_posts([make_post("100")])
    original = store.posts_path.read_bytes()
    real_replace = os.replace
    interrupted = False

    def replace_then_interrupt(source, target):
        nonlocal interrupted
        if Path(target) == store.posts_path and not interrupted:
            interrupted = True
            real_replace(source, target)
            raise KeyboardInterrupt("after target replace")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="target"):
        store.save_posts([make_post("101")])

    assert store.posts_path.read_bytes() == original
    assert [post.id for post in store.read_posts()] == ["100"]
    assert not store.transaction_manifest_path.exists()


def test_second_keyboard_interrupt_during_rollback_is_recovered_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    store = FilePostStore(data_dir)
    store.save_posts([make_post("100")])
    original = store.posts_path.read_bytes()
    real_replace = os.replace
    post_replacements = 0

    def interrupt_target_then_rollback(source, target):
        nonlocal post_replacements
        if Path(target) == store.posts_path:
            post_replacements += 1
            if post_replacements == 1:
                real_replace(source, target)
                raise KeyboardInterrupt("after target replace")
            if post_replacements == 2:
                raise KeyboardInterrupt("during rollback")
        return real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt_target_then_rollback)
        with pytest.raises(KeyboardInterrupt, match="during rollback"):
            store.save_posts([make_post("101")])

    assert store.transaction_manifest_path.is_file()
    assert store.posts_path.read_bytes() != original

    recovered = FilePostStore(data_dir)

    assert recovered.posts_path.read_bytes() == original
    assert [post.id for post in recovered.read_posts()] == ["100"]
    assert not recovered.transaction_manifest_path.exists()
    assert not list(data_dir.rglob("*.backup.tmp"))


def test_file_store_rejects_manifest_target_outside_managed_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = FilePostStore(data_dir)
    sentinel = data_dir / "sentinel.txt"
    sentinel.write_text("keep-me", encoding="utf-8")
    temporary = data_dir / ".sentinel.txt.simulated.tmp"
    temporary.write_text("replacement", encoding="utf-8")
    store.transaction_manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "target": "sentinel.txt",
                        "temporary": temporary.name,
                        "backup": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StorageCorruptionError, match="transaction target"):
        FilePostStore(data_dir)

    assert sentinel.read_text(encoding="utf-8") == "keep-me"
    assert temporary.read_text(encoding="utf-8") == "replacement"


def test_file_store_rejects_manifest_path_role_aliasing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = FilePostStore(data_dir)
    store.save_posts([make_post("100")])
    original = store.posts_path.read_bytes()
    alias = store.posts_path.parent / f".{store.posts_path.name}.alias.backup.tmp"
    shutil.copyfile(store.posts_path, alias)
    store.transaction_manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "target": "posts/posts.jsonl",
                        "temporary": "posts/" + alias.name,
                        "backup": "posts/" + alias.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StorageCorruptionError, match="temporary path|unique"):
        FilePostStore(data_dir)

    assert store.posts_path.read_bytes() == original


def test_file_store_never_silently_overwrites_damaged_json(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.sync_states_path.write_text('{"list-a":', encoding="utf-8")
    original = store.sync_states_path.read_bytes()

    with pytest.raises(StorageCorruptionError, match="invalid JSON"):
        store.save_sync_state(make_state())

    assert store.sync_states_path.read_bytes() == original


def test_file_store_treats_empty_sync_json_as_corruption(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.sync_states_path.write_bytes(b"")

    with pytest.raises(StorageCorruptionError, match="empty JSON"):
        store.save_sync_state(make_state())

    assert store.sync_states_path.read_bytes() == b""


def test_file_store_rejects_duplicate_records_in_existing_jsonl(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_posts([make_post("100")])
    line = store.posts_path.read_text(encoding="utf-8")
    store.posts_path.write_text(line + line, encoding="utf-8")

    with pytest.raises(StorageCorruptionError, match="duplicate id"):
        store.has_post("100")


def test_file_store_rejects_orphan_membership(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.memberships_path.write_text(
        '{"list_id":"list-a","post_id":"missing"}\n', encoding="utf-8"
    )

    with pytest.raises(StorageCorruptionError, match="missing post"):
        store.has_membership("missing", "list-a")
    with pytest.raises(StorageCorruptionError, match="missing posts"):
        store.save_posts([make_post("100")])


@pytest.mark.parametrize(
    "invalid_json",
    [
        '{"id":"100","id":"101"}\n',
        '{"id":"100","raw_payload":{"score":NaN}}\n',
    ],
)
def test_file_store_rejects_non_strict_jsonl(
    tmp_path: Path, invalid_json: str
) -> None:
    store = FilePostStore(tmp_path / "data")
    store.posts_path.write_text(invalid_json, encoding="utf-8")

    with pytest.raises(StorageCorruptionError, match="invalid JSONL"):
        store.read_posts()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("references", "broken"),
        ("urls", {"expanded_url": "https://example.test"}),
        ("raw_payload", []),
        ("metrics", {"likes": "many"}),
    ],
)
def test_file_store_rejects_valid_json_with_damaged_post_schema(
    tmp_path: Path, field: str, invalid_value
) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_posts([make_post("100")])
    payload = json.loads(store.posts_path.read_text(encoding="utf-8"))
    payload[field] = invalid_value
    store.posts_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    damaged = store.posts_path.read_bytes()

    with pytest.raises(StorageCorruptionError, match="invalid XPost"):
        store.save_posts([make_post("101")])

    assert store.posts_path.read_bytes() == damaged


def test_file_store_rejects_valid_json_with_damaged_state_schema(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_sync_state(make_state())
    payload = json.loads(store.sync_states_path.read_text(encoding="utf-8"))
    payload["list-a"]["last_attempt_at"] = 123
    store.sync_states_path.write_text(json.dumps(payload), encoding="utf-8")
    damaged = store.sync_states_path.read_bytes()

    with pytest.raises(StorageCorruptionError, match="ListSyncState"):
        store.save_sync_state(make_state(newest_post_id="200"))

    assert store.sync_states_path.read_bytes() == damaged


def test_file_store_rejects_valid_json_with_damaged_run_schema(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.create_run(make_run())
    path = store.runs_dir / "run_run-1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["requested_lists"] = "list-a"
    path.write_text(json.dumps(payload), encoding="utf-8")
    damaged = path.read_bytes()

    with pytest.raises(StorageCorruptionError, match="IngestionRun"):
        store.update_run(make_run(status="failed"))

    assert path.read_bytes() == damaged


def test_invalid_post_is_rejected_before_existing_file_changes(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_posts([make_post("100")])
    original = store.posts_path.read_bytes()
    invalid = make_post("101")
    invalid.raw_payload = {"not-json": object()}

    with pytest.raises(StorageValidationError, match="invalid XPost"):
        store.save_posts([invalid])

    assert store.posts_path.read_bytes() == original
    assert not store.has_post("101")


def test_mismatched_embedded_author_is_rejected_without_writes(store) -> None:
    invalid = make_post("101")
    invalid.author_id = "different-author"

    with pytest.raises(StorageValidationError, match="must match"):
        store.save_posts([invalid])

    assert not store.has_post("101")


def test_atomic_replace_failure_preserves_old_target_and_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_posts([make_post("100")])
    original = store.posts_path.read_bytes()
    real_replace = os.replace

    def fail_post_replace(source, target):
        if Path(target) == store.posts_path:
            raise OSError("injected replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_post_replace)
    with pytest.raises(OSError, match="injected"):
        store.save_posts([make_post("101")])

    assert store.posts_path.read_bytes() == original
    assert list(store.data_dir.rglob("*.tmp")) == []


def test_entity_replace_failure_does_not_advance_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FilePostStore(tmp_path / "data")
    store.commit_list_batch([make_post("100")], "list-a", make_state())
    real_replace = os.replace

    def fail_author_replace(source, target):
        if Path(target) == store.authors_path:
            raise OSError("injected author failure")
        return real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_author_replace)
        with pytest.raises(OSError, match="injected"):
            store.commit_list_batch(
                [make_post("101", author_id="author-2")],
                "list-a",
                make_state(newest_post_id="101"),
            )

    # Catchable multi-file failures restore the complete old snapshot.
    assert not store.has_post("101")
    assert not store.has_membership("101", "list-a")
    assert store.get_sync_state("list-a").newest_post_id == "100"

    # Retrying the same idempotent batch completes the missing relation/state.
    store.commit_list_batch(
        [make_post("101", author_id="author-2")],
        "list-a",
        make_state(newest_post_id="101"),
    )
    assert store.has_membership("101", "list-a")
    assert store.get_sync_state("list-a").newest_post_id == "101"


def test_two_file_store_instances_share_a_thread_lock(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    first = FilePostStore(data_dir)
    second = FilePostStore(data_dir)

    def save(index: int) -> None:
        target = first if index % 2 else second
        target.save_posts(
            [
                make_post(
                    str(index),
                    list_id=f"list-{index % 3}",
                    author_id=f"author-{index}",
                )
            ]
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(save, range(1, 13)))

    assert len(first.read_posts()) == 12
    assert len(second.read_memberships()) == 12
    assert len(first.read_authors()) == 12


def test_file_store_process_lock_blocks_cross_process_write(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_posts([make_post("100")])
    started = tmp_path / "child-started"
    finished = tmp_path / "child-finished"
    child = """
import sys
from datetime import datetime, timezone
from pathlib import Path
from app.models import XPost
from app.storage import FilePostStore

data_dir, started_path, finished_path = sys.argv[1:4]
Path(started_path).write_text("started", encoding="utf-8")
post = XPost(
    id="101", text="child", author_id="", author=None,
    created_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    language="en", conversation_id="101", source_list_id="list-a",
    raw_payload={"id": "101"},
    fetched_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
)
FilePostStore(data_dir).save_posts([post])
Path(finished_path).write_text("finished", encoding="utf-8")
"""

    with store._guard():
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                child,
                str(store.data_dir),
                str(started),
                str(finished),
            ],
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert started.exists(), "child process did not reach storage write"
        time.sleep(0.3)
        assert not finished.exists(), "child bypassed the cross-process lock"

    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, (stdout, stderr)
    assert finished.exists(), "child did not resume after lock release"
    assert store.has_post("101")


def test_file_and_memory_store_export_equivalent_data(tmp_path: Path) -> None:
    file_store = FilePostStore(tmp_path / "data")
    memory_store = InMemoryPostStore()
    posts = [make_post("100"), make_post("100", list_id="list-b"), make_post("101")]
    for store in (file_store, memory_store):
        store.save_posts(posts)
        store.save_sync_state(make_state())
        store.create_run(make_run())

    assert file_store.export_data() == memory_store.export_data()


def test_existing_file_changes_invalidate_fast_indexes(tmp_path: Path) -> None:
    store = FilePostStore(tmp_path / "data")
    store.save_posts([make_post("100")])
    assert store.has_post("100")

    # A second store atomically replaces the file.  The first store notices
    # the file signature change instead of returning a stale cached answer.
    other = FilePostStore(store.data_dir)
    other.save_posts([make_post("101")])

    assert store.has_post("101")
