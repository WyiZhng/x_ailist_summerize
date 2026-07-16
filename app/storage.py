"""Thread-safe file and in-memory persistence for normalized X ingestion data.

The file store deliberately keeps the durable ingestion records separate from
HTML report history.  Entity files are rewritten through temporary files and
``os.replace`` so an interrupted write never exposes a partially written JSON
document.  A list checkpoint is always replaced *after* its posts, authors and
memberships; consequently a failed entity write cannot advance incremental
sync state.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Protocol, TypeVar, runtime_checkable

from .models import IngestionRun, ListSyncState, XAuthor, XPost, ensure_aware_datetime
from .security import redact_sensitive_text


logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Base class for durable ingestion storage errors."""


class StorageCorruptionError(StorageError):
    """Raised when an existing storage file is malformed or inconsistent."""


class StorageValidationError(StorageError, ValueError):
    """Raised before writing data that does not match the storage schema."""


class StorageConflictError(StorageError):
    """Raised when another collector advanced a List checkpoint concurrently."""


class StorageOwnershipConflictError(StorageConflictError):
    """Raised when a stale ingestion owner attempts to mutate a reclaimed run."""


_Model = TypeVar("_Model", XPost, XAuthor, ListSyncState, IngestionRun)
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CHECKPOINT_UNSET = object()
_SENSITIVE_ERROR_KEYS = frozenset(
    {
        "api_key",
        "api-key",
        "authorization",
        "auth_token",
        "auth-token",
        "bearer_token",
        "bearer-token",
        "cookie",
        "cookies",
        "password",
        "secret",
        "token",
    }
)


@runtime_checkable
class PostStore(Protocol):
    """Persistence contract used by the incremental ingestion workflow."""

    def has_post(self, post_id: str) -> bool:
        """Return whether the globally unique post entity already exists."""

    def has_membership(self, post_id: str, list_id: str) -> bool:
        """Return whether a post has already been observed in this X List."""

    def save_posts(self, posts: Sequence[XPost]) -> None:
        """Idempotently persist posts, embedded authors and source memberships."""

    def commit_list_batch(
        self,
        posts: Sequence[XPost],
        list_id: str,
        state: ListSyncState,
        *,
        expected_newest_post_id: str | None | object = ...,
        run: IngestionRun | None = None,
    ) -> set[str]:
        """Persist one list batch if its previously read checkpoint is current."""

    def get_sync_state(self, list_id: str) -> ListSyncState | None:
        """Read the independent checkpoint for one X List."""

    def save_sync_state(
        self,
        state: ListSyncState,
        *,
        expected_newest_post_id: str | None | object = ...,
    ) -> None:
        """Persist state, optionally requiring an unchanged checkpoint."""

    def create_run(self, run: IngestionRun) -> None:
        """Create a new ingestion run record."""

    def update_run(
        self,
        run: IngestionRun,
        *,
        expected_report_status: str | object = ...,
        expected_report_claim_id: str | None | object = ...,
    ) -> None:
        """Replace a run, optionally using report claim fields as CAS guards."""

    def update_ingestion_run(
        self,
        run: IngestionRun,
        *,
        expected_ingestion_owner_id: str | None | object = ...,
        expected_ingestion_heartbeat_at: datetime | None | object = ...,
    ) -> None:
        """Update ingestion fields with optional owner/heartbeat CAS guards."""

    def get_run(self, run_id: str) -> IngestionRun | None:
        """Read an ingestion run by ID."""

    def read_posts(self, list_id: str | None = None) -> list[XPost]:
        """Read all posts, optionally restricted by list membership."""

    def export_data(self) -> dict[str, Any]:
        """Return JSON-compatible data suitable for a future DB migration."""


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StorageValidationError(f"{field} must be a non-empty string")
    return value


def _validate_run_id(run_id: Any) -> str:
    value = _require_identifier(run_id, "run_id")
    if value in {".", ".."} or not _RUN_ID_RE.fullmatch(value):
        raise StorageValidationError(
            "run_id must contain only letters, digits, dots, underscores or hyphens"
        )
    return value


def _validate_post_identity(post: XPost) -> None:
    _require_identifier(post.id, "post_id")
    _require_identifier(post.source_list_id, "source_list_id")
    if post.author is not None:
        embedded_id = _require_identifier(post.author.id, "author.id")
        author_id = _require_identifier(post.author_id, "author_id")
        if embedded_id != author_id:
            raise StorageValidationError("post author_id must match embedded author.id")
    for reference in post.references:
        _require_identifier(reference.relation_type, "reference.relation_type")
        _require_identifier(reference.referenced_post_id, "reference.referenced_post_id")


def _model_payload(value: _Model, model_type: type[_Model]) -> dict[str, Any]:
    """Validate a model and return a JSON-safe canonical dictionary."""
    if not isinstance(value, model_type):
        raise StorageValidationError(
            f"expected {model_type.__name__}, got {type(value).__name__}"
        )
    try:
        payload = value.to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("to_dict() did not return an object")
        # Reject NaN/Infinity and values that cannot be represented by JSON.
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        validated = model_type.from_dict(dict(payload))
        canonical = validated.to_dict()
        json.dumps(canonical, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StorageValidationError(f"invalid {model_type.__name__}: {exc}") from exc
    return dict(canonical)


def _expect_stored_type(
    payload: Mapping[str, Any],
    field: str,
    expected: type[Any] | tuple[type[Any], ...],
    *,
    required: bool = False,
) -> Any:
    if field not in payload:
        if required:
            raise ValueError(f"missing required field {field}")
        return None
    value = payload[field]
    if not isinstance(value, expected):
        names = (
            ", ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise TypeError(f"{field} must be {names}")
    return value


def _expect_optional_stored_string(
    payload: Mapping[str, Any], field: str, *, required: bool = False
) -> None:
    if field not in payload:
        if required:
            raise ValueError(f"missing required field {field}")
        return
    if payload[field] is not None and not isinstance(payload[field], str):
        raise TypeError(f"{field} must be string or null")


def _expect_stored_count(payload: Mapping[str, Any], field: str) -> None:
    if field not in payload:
        return
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field} must be a non-negative integer")


def _validate_author_payload(payload: Mapping[str, Any]) -> None:
    author_id = _expect_stored_type(payload, "id", str, required=True)
    _require_identifier(author_id, "author.id")
    if "username" in payload and not isinstance(payload["username"], str):
        raise TypeError("username must be string")
    _expect_optional_stored_string(payload, "name")
    _expect_optional_stored_string(payload, "profile_image_url")


def _validate_post_payload(payload: Mapping[str, Any]) -> None:
    post_id = _expect_stored_type(payload, "id", str, required=True)
    _require_identifier(post_id, "post_id")
    if "text" in payload and not isinstance(payload["text"], str):
        raise TypeError("text must be string")
    if "author_id" in payload and not isinstance(payload["author_id"], str):
        raise TypeError("author_id must be string")
    author = payload.get("author")
    if author is not None:
        if not isinstance(author, Mapping):
            raise TypeError("author must be object or null")
        _validate_author_payload(author)
    for field in ("created_at", "language", "conversation_id"):
        _expect_optional_stored_string(payload, field)
    source_list_id = _expect_stored_type(
        payload, "source_list_id", str, required=True
    )
    _require_identifier(source_list_id, "source_list_id")

    metrics = payload.get("metrics")
    if metrics is not None:
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be object")
        for field in ("likes", "retweets", "replies", "quotes", "bookmarks"):
            _expect_stored_count(metrics, field)
        if "impressions" in metrics and metrics["impressions"] is not None:
            _expect_stored_count(metrics, "impressions")

    references = payload.get("references")
    if references is not None:
        if not isinstance(references, list):
            raise TypeError("references must be array")
        for index, reference in enumerate(references):
            if not isinstance(reference, Mapping):
                raise TypeError(f"references[{index}] must be object")
            relation = _expect_stored_type(
                reference, "relation_type", str, required=True
            )
            target = _expect_stored_type(
                reference, "referenced_post_id", str, required=True
            )
            _require_identifier(relation, f"references[{index}].relation_type")
            _require_identifier(target, f"references[{index}].referenced_post_id")

    urls = payload.get("urls")
    if urls is not None and (
        not isinstance(urls, list) or not all(isinstance(item, str) for item in urls)
    ):
        raise TypeError("urls must be an array of strings")
    media = payload.get("media")
    if media is not None and (
        not isinstance(media, list) or not all(isinstance(item, Mapping) for item in media)
    ):
        raise TypeError("media must be an array of objects")
    if "raw_payload" in payload and not isinstance(payload["raw_payload"], Mapping):
        raise TypeError("raw_payload must be object")
    _expect_optional_stored_string(payload, "fetched_at")


def _validate_sync_state_payload(payload: Mapping[str, Any]) -> None:
    list_id = _expect_stored_type(payload, "list_id", str, required=True)
    _require_identifier(list_id, "list_id")
    for field in ("newest_post_id", "last_run_id", "last_error"):
        _expect_optional_stored_string(payload, field)
    for field in (
        "newest_post_created_at",
        "last_attempt_at",
        "last_success_at",
    ):
        _expect_optional_stored_string(payload, field)
    if "last_status" in payload and not isinstance(payload["last_status"], str):
        raise TypeError("last_status must be string")


def _validate_ingestion_run_payload(payload: Mapping[str, Any]) -> None:
    run_id = _expect_stored_type(payload, "run_id", str, required=True)
    _validate_run_id(run_id)
    _expect_stored_type(payload, "started_at", str, required=True)
    _expect_optional_stored_string(payload, "finished_at")
    _expect_optional_stored_string(payload, "ingestion_owner_id")
    _expect_optional_stored_string(payload, "ingestion_heartbeat_at")
    if "status" in payload and not isinstance(payload["status"], str):
        raise TypeError("status must be string")
    for field in (
        "requested_lists",
        "successful_lists",
        "failed_lists",
        "new_post_ids",
    ):
        if field in payload:
            value = payload[field]
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise TypeError(f"{field} must be an array of strings")
    for field in ("fetched_count", "new_post_count", "duplicate_count"):
        _expect_stored_count(payload, field)
    if "errors" in payload and (
        not isinstance(payload["errors"], list)
        or not all(isinstance(item, Mapping) for item in payload["errors"])
    ):
        raise TypeError("errors must be an array of objects")
    if "report_status" in payload and not isinstance(payload["report_status"], str):
        raise TypeError("report_status must be string")
    _expect_optional_stored_string(payload, "report_error")
    _expect_optional_stored_string(payload, "report_claim_id")
    _expect_optional_stored_string(payload, "report_claimed_at")


def _validate_stored_model_payload(
    payload: Mapping[str, Any], model_type: type[_Model]
) -> None:
    """Reject wrong persisted field types before tolerant model normalization."""
    if model_type is XPost:
        _validate_post_payload(payload)
    elif model_type is XAuthor:
        _validate_author_payload(payload)
    elif model_type is ListSyncState:
        _validate_sync_state_payload(payload)
    elif model_type is IngestionRun:
        _validate_ingestion_run_payload(payload)


def _clone_model(value: _Model, model_type: type[_Model]) -> _Model:
    return model_type.from_dict(_model_payload(value, model_type))


def _redact_error_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact common credentials in persisted error metadata."""
    normalized_key = key.strip().lower() if isinstance(key, str) else None
    if normalized_key in _SENSITIVE_ERROR_KEYS:
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_error_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_error_value(item) for item in value]
    return value


def _safe_sync_state(value: ListSyncState) -> ListSyncState:
    clone = _clone_model(value, ListSyncState)
    if clone.last_error is not None:
        clone.last_error = redact_sensitive_text(clone.last_error)
    return clone


def _safe_ingestion_run(value: IngestionRun) -> IngestionRun:
    clone = _clone_model(value, IngestionRun)
    clone.errors = [_redact_error_value(error) for error in clone.errors]
    if clone.report_error is not None:
        clone.report_error = redact_sensitive_text(clone.report_error)
    return clone


def _preserve_report_state(target: IngestionRun, current: IngestionRun) -> None:
    """Keep report-owned fields when a stale ingestion snapshot is merged."""

    target.report_status = current.report_status
    target.report_error = current.report_error
    target.report_claim_id = current.report_claim_id
    target.report_claimed_at = current.report_claimed_at


def _json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _strict_json_loads(text: str) -> Any:
    """Decode RFC-compliant JSON, rejecting constants and duplicate keys."""

    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-standard JSON constant {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key {key!r}")
            value[key] = child
        return value

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


class _ProcessLockState:
    """Reentrant process-lock state shared by same-path store instances."""

    __slots__ = ("depth", "handle", "pid")

    def __init__(self) -> None:
        self.depth = 0
        self.handle: BinaryIO | None = None
        self.pid: int | None = None


class FilePostStore:
    """JSONL/JSON-backed ingestion store with thread/process serialization.

    Locks are shared by all ``FilePostStore`` instances targeting the same
    resolved directory.  An exclusive advisory lock on ``.store.lock`` also
    serializes Web/CLI processes.  Readers take the same lock as writers so no
    process can observe the posts/authors/memberships/checkpoint replacement
    sequence halfway through.

    Catchable write failures roll back all files to their pre-call snapshots.
    A durable transaction manifest lets the next process perform the same
    rollback after a hard process/OS crash.  The checkpoint is also replaced
    last, providing a second replay boundary for incremental ingestion.
    """

    _locks_guard = threading.Lock()
    _path_locks: dict[str, threading.RLock] = {}
    _process_states: dict[str, _ProcessLockState] = {}

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.posts_path = self.data_dir / "posts" / "posts.jsonl"
        self.authors_path = self.data_dir / "authors" / "authors.jsonl"
        self.memberships_path = (
            self.data_dir / "list_memberships" / "post_lists.jsonl"
        )
        self.sync_states_path = self.data_dir / "sync" / "list_sync_states.json"
        self.runs_dir = self.data_dir / "runs"
        self.process_lock_path = self.data_dir / ".store.lock"
        self.transaction_manifest_path = self.data_dir / ".store.transaction.json"

        lock_key = os.path.normcase(str(self.data_dir.resolve(strict=False)))
        with self._locks_guard:
            self._lock = self._path_locks.setdefault(lock_key, threading.RLock())
            self._process_state = self._process_states.setdefault(
                lock_key, _ProcessLockState()
            )

        self._post_ids: set[str] | None = None
        self._post_signature: tuple[int, int, int, int] | None = None
        self._membership_ids: set[tuple[str, str]] | None = None
        self._membership_signature: tuple[int, int, int, int] | None = None
        self._ensure_layout()
        with self._guard():
            # The outermost guard performs recovery after it owns the process
            # lock.  Enter once during construction so startup still validates
            # any manifest left by a crashed writer.
            pass

    def _acquire_process_lock(self) -> None:
        state = self._process_state
        current_pid = os.getpid()
        if state.depth and state.pid == current_pid:
            state.depth += 1
            return
        if state.pid is not None and state.pid != current_pid:
            # A POSIX fork copies Python lock bookkeeping.  Discard the
            # child's inherited descriptor/state and acquire a fresh OS lock.
            if state.handle is not None:
                try:
                    state.handle.close()
                except OSError:
                    pass
            state.depth = 0
            state.handle = None
            state.pid = None

        handle: BinaryIO | None = None
        try:
            handle = self.process_lock_path.open("a+b", buffering=0)
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            raise StorageError(
                f"cannot acquire ingestion storage lock {self.process_lock_path}: {exc}"
            ) from exc
        state.handle = handle
        state.depth = 1
        state.pid = current_pid

    def _release_process_lock(self) -> None:
        state = self._process_state
        if state.depth <= 0 or state.handle is None or state.pid != os.getpid():
            raise StorageError("ingestion storage lock state is inconsistent")
        state.depth -= 1
        if state.depth:
            return

        handle = state.handle
        state.handle = None
        state.pid = None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise StorageError(
                f"cannot release ingestion storage lock {self.process_lock_path}: {exc}"
            ) from exc
        finally:
            handle.close()

    @contextmanager
    def _guard(self) -> Iterator[None]:
        """Take the shared reentrant thread lock and exclusive process lock."""
        with self._lock:
            self._acquire_process_lock()
            try:
                # An already-created store in another process must be able to
                # take over immediately after a writer hard-crashes.  Only the
                # outermost acquisition performs recovery: nested guarded
                # reads during an active transaction must not roll that live
                # transaction back.
                if self._process_state.depth == 1:
                    self._recover_interrupted_transaction()
                yield
            finally:
                self._release_process_lock()

    def _ensure_layout(self) -> None:
        for directory in (
            self.posts_path.parent,
            self.authors_path.parent,
            self.memberships_path.parent,
            self.sync_states_path.parent,
            self.runs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int, int] | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StorageError(f"cannot inspect storage file {path}: {exc}") from exc
        return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)

    def _invalidate_indexes(self) -> None:
        self._post_ids = None
        self._post_signature = None
        self._membership_ids = None
        self._membership_signature = None

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except (OSError, UnicodeError) as exc:
            raise StorageCorruptionError(f"cannot read UTF-8 storage file {path}: {exc}") from exc

    def _read_model_jsonl(
        self,
        path: Path,
        model_type: type[_Model],
        id_field: str,
    ) -> dict[str, _Model]:
        text = self._read_text(path)
        if not text:
            return {}
        lines = text.splitlines()
        values: dict[str, _Model] = {}
        for index, line in enumerate(lines, start=1):
            suffix = " (final line)" if index == len(lines) else ""
            if not line.strip():
                raise StorageCorruptionError(
                    f"invalid JSONL in {path} at line {index}{suffix}: blank line"
                )
            try:
                payload = _strict_json_loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise StorageCorruptionError(
                    f"invalid JSONL in {path} at line {index}{suffix}: "
                    f"{getattr(exc, 'msg', str(exc))}"
                ) from exc
            if not isinstance(payload, dict):
                raise StorageCorruptionError(
                    f"invalid JSONL in {path} at line {index}{suffix}: expected object"
                )
            try:
                _validate_stored_model_payload(payload, model_type)
                value = model_type.from_dict(payload)
                canonical = _model_payload(value, model_type)
                identifier = _require_identifier(canonical.get(id_field), id_field)
                if model_type is XPost:
                    _validate_post_identity(value)
            except (StorageValidationError, TypeError, ValueError) as exc:
                raise StorageCorruptionError(
                    f"invalid {model_type.__name__} in {path} at line {index}{suffix}: {exc}"
                ) from exc
            if identifier in values:
                raise StorageCorruptionError(
                    f"duplicate {id_field} {identifier!r} in {path} at line {index}"
                )
            values[identifier] = value
        return values

    def _read_memberships(self) -> set[tuple[str, str]]:
        path = self.memberships_path
        text = self._read_text(path)
        if not text:
            return set()
        lines = text.splitlines()
        memberships: set[tuple[str, str]] = set()
        for index, line in enumerate(lines, start=1):
            suffix = " (final line)" if index == len(lines) else ""
            if not line.strip():
                raise StorageCorruptionError(
                    f"invalid JSONL in {path} at line {index}{suffix}: blank line"
                )
            try:
                payload = _strict_json_loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise StorageCorruptionError(
                    f"invalid JSONL in {path} at line {index}{suffix}: "
                    f"{getattr(exc, 'msg', str(exc))}"
                ) from exc
            if not isinstance(payload, dict) or set(payload) != {"post_id", "list_id"}:
                raise StorageCorruptionError(
                    f"invalid membership in {path} at line {index}{suffix}: "
                    "expected exactly post_id and list_id"
                )
            try:
                key = (
                    _require_identifier(payload["post_id"], "post_id"),
                    _require_identifier(payload["list_id"], "list_id"),
                )
            except StorageValidationError as exc:
                raise StorageCorruptionError(
                    f"invalid membership in {path} at line {index}{suffix}: {exc}"
                ) from exc
            if key in memberships:
                raise StorageCorruptionError(
                    f"duplicate membership {key!r} in {path} at line {index}"
                )
            memberships.add(key)
        return memberships

    @staticmethod
    def _validate_membership_targets(
        memberships: Iterable[tuple[str, str]], post_ids: Iterable[str]
    ) -> None:
        known = set(post_ids)
        missing = sorted({post_id for post_id, _ in memberships if post_id not in known})
        if missing:
            raise StorageCorruptionError(
                f"memberships reference missing posts: {missing!r}"
            )

    def _read_sync_states(self) -> dict[str, ListSyncState]:
        path = self.sync_states_path
        if not path.exists():
            return {}
        text = self._read_text(path)
        if not text:
            raise StorageCorruptionError(f"empty JSON storage file {path}")
        try:
            payload = _strict_json_loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise StorageCorruptionError(
                f"invalid JSON in {path}: {getattr(exc, 'msg', str(exc))}"
            ) from exc
        if not isinstance(payload, dict):
            raise StorageCorruptionError(f"invalid JSON in {path}: expected object")
        states: dict[str, ListSyncState] = {}
        for list_id, raw in payload.items():
            try:
                key = _require_identifier(list_id, "list_id")
                if not isinstance(raw, dict):
                    raise TypeError("expected object")
                _validate_stored_model_payload(raw, ListSyncState)
                state = ListSyncState.from_dict(raw)
                canonical = _model_payload(state, ListSyncState)
                if canonical.get("list_id") != key:
                    raise ValueError("object list_id does not match its map key")
            except (StorageValidationError, TypeError, ValueError) as exc:
                raise StorageCorruptionError(
                    f"invalid ListSyncState {list_id!r} in {path}: {exc}"
                ) from exc
            states[key] = state
        return states

    def _read_run_path(self, path: Path) -> IngestionRun:
        text = self._read_text(path)
        if not text:
            raise StorageCorruptionError(f"empty ingestion run file {path}")
        try:
            payload = _strict_json_loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise StorageCorruptionError(
                f"invalid JSON in {path}: {getattr(exc, 'msg', str(exc))}"
            ) from exc
        if not isinstance(payload, dict):
            raise StorageCorruptionError(f"invalid JSON in {path}: expected object")
        try:
            expected_id = _validate_run_id(path.name[len("run_") : -len(".json")])
            _validate_stored_model_payload(payload, IngestionRun)
            run = IngestionRun.from_dict(payload)
            canonical = _model_payload(run, IngestionRun)
            if canonical.get("run_id") != expected_id:
                raise ValueError("run_id does not match its filename")
        except (StorageValidationError, TypeError, ValueError) as exc:
            raise StorageCorruptionError(f"invalid IngestionRun in {path}: {exc}") from exc
        return run

    @staticmethod
    def _render_model_jsonl(values: Mapping[str, _Model], model_type: type[_Model]) -> str:
        return "".join(
            f"{_json_line(_model_payload(values[key], model_type))}\n"
            for key in sorted(values)
        )

    @staticmethod
    def _render_memberships(values: Iterable[tuple[str, str]]) -> str:
        return "".join(
            f"{_json_line({'post_id': post_id, 'list_id': list_id})}\n"
            for post_id, list_id in sorted(values, key=lambda item: (item[1], item[0]))
        )

    @staticmethod
    def _render_sync_states(values: Mapping[str, ListSyncState]) -> str:
        payload = {
            key: _model_payload(values[key], ListSyncState) for key in sorted(values)
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"

    @staticmethod
    def _render_run(run: IngestionRun) -> str:
        return json.dumps(
            _model_payload(run, IngestionRun),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        # File fsync + atomic replace is the portable guarantee.  Directory
        # fsync additionally hardens POSIX durability; Windows does not expose
        # a portable directory descriptor, so unsupported errors are ignored.
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except OSError as exc:
            if os.name == "nt":
                return
            raise StorageError(f"cannot open storage directory for fsync: {directory}") from exc
        try:
            os.fsync(descriptor)
        except OSError as exc:
            unsupported = {
                errno.EBADF,
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if os.name != "nt" and exc.errno not in unsupported:
                raise StorageError(
                    f"cannot fsync storage directory: {directory}"
                ) from exc
        finally:
            os.close(descriptor)

    def _manifest_relative_path(self, path: Path) -> str:
        root = self.data_dir.resolve(strict=False)
        resolved = path.resolve(strict=False)
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise StorageValidationError(
                f"transaction path escapes storage directory: {path}"
            ) from exc
        return relative.as_posix()

    def _manifest_absolute_path(self, value: Any) -> Path:
        if not isinstance(value, str) or not value:
            raise StorageCorruptionError("invalid storage transaction path")
        root = self.data_dir.resolve(strict=False)
        resolved = (root / Path(value)).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise StorageCorruptionError(
                "storage transaction path escapes the data directory"
            ) from exc
        return resolved

    def _is_managed_transaction_target(self, target: Path) -> bool:
        resolved = target.resolve(strict=False)
        fixed_targets = {
            self.posts_path.resolve(strict=False),
            self.authors_path.resolve(strict=False),
            self.memberships_path.resolve(strict=False),
            self.sync_states_path.resolve(strict=False),
        }
        if resolved in fixed_targets:
            return True
        if resolved.parent != self.runs_dir.resolve(strict=False):
            return False
        name = resolved.name
        if not name.startswith("run_") or not name.endswith(".json"):
            return False
        try:
            _validate_run_id(name[len("run_") : -len(".json")])
        except StorageValidationError:
            return False
        return True

    def _write_transaction_manifest(
        self,
        staged: Sequence[tuple[Path, Path, Path | None]],
    ) -> None:
        payload = {
            "version": 1,
            "entries": [
                {
                    "target": self._manifest_relative_path(target),
                    "temporary": self._manifest_relative_path(temporary),
                    "backup": (
                        self._manifest_relative_path(backup)
                        if backup is not None
                        else None
                    ),
                }
                for target, temporary, backup in staged
            ],
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".store.transaction.", suffix=".tmp", dir=self.data_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.transaction_manifest_path)
            self._fsync_directory(self.data_dir)
        except BaseException:
            self._cleanup_transaction_file(temporary)
            raise

    def _clear_transaction_manifest(self) -> None:
        self.transaction_manifest_path.unlink(missing_ok=True)
        self._fsync_directory(self.data_dir)

    @staticmethod
    def _cleanup_transaction_file(path: Path) -> None:
        """Best-effort cleanup after the manifest no longer owns durability."""

        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Once the manifest is cleared, backup/temp cleanup must not turn
            # a committed (or fully rolled-back) transaction into an apparent
            # business failure.  A later maintenance pass may remove orphans.
            logger.warning("storage_transaction_cleanup_failed file=%s", path.name)

    def _restore_backup(self, backup: Path, target: Path) -> None:
        if not backup.is_file():
            raise StorageCorruptionError(
                f"storage transaction backup is missing: {backup}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.restore.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                with backup.open("rb") as source:
                    shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, target)
            self._fsync_directory(target.parent)
        finally:
            self._cleanup_transaction_file(temporary)

    def _recover_interrupted_transaction(self) -> None:
        path = self.transaction_manifest_path
        if not path.exists():
            return
        text = self._read_text(path)
        try:
            payload = _strict_json_loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise StorageCorruptionError(
                f"invalid storage transaction manifest {path}: {exc}"
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or not isinstance(payload.get("entries"), list)
        ):
            raise StorageCorruptionError(
                f"invalid storage transaction manifest {path}: unexpected schema"
            )

        entries: list[tuple[Path, Path, Path | None]] = []
        targets: set[Path] = set()
        role_paths: set[Path] = set()
        for raw in payload["entries"]:
            if not isinstance(raw, dict) or set(raw) != {
                "target",
                "temporary",
                "backup",
            }:
                raise StorageCorruptionError(
                    f"invalid storage transaction manifest {path}: invalid entry"
                )
            target = self._manifest_absolute_path(raw["target"])
            temporary = self._manifest_absolute_path(raw["temporary"])
            backup = (
                self._manifest_absolute_path(raw["backup"])
                if raw["backup"] is not None
                else None
            )
            if not self._is_managed_transaction_target(target):
                raise StorageCorruptionError(
                    f"invalid storage transaction target: {target}"
                )
            if target in targets:
                raise StorageCorruptionError(
                    f"duplicate storage transaction target: {target}"
                )
            if temporary.parent != target.parent or not (
                temporary.name.startswith(f".{target.name}.")
                and temporary.name.endswith(".tmp")
                and not temporary.name.endswith(".backup.tmp")
            ):
                raise StorageCorruptionError(
                    f"invalid storage transaction temporary path: {temporary}"
                )
            paths_for_entry = {target, temporary}
            if backup is not None:
                if backup.parent != target.parent or not (
                    backup.name.startswith(f".{target.name}.")
                    and backup.name.endswith(".backup.tmp")
                ):
                    raise StorageCorruptionError(
                        f"invalid storage transaction backup path: {backup}"
                    )
                if not backup.is_file():
                    raise StorageCorruptionError(
                        f"storage transaction backup is missing: {backup}"
                    )
                paths_for_entry.add(backup)
            expected_role_count = 3 if backup is not None else 2
            if len(paths_for_entry) != expected_role_count or role_paths.intersection(
                paths_for_entry
            ):
                raise StorageCorruptionError(
                    "storage transaction paths must be unique across all roles"
                )
            targets.add(target)
            role_paths.update(paths_for_entry)
            entries.append((target, temporary, backup))

        for target, _, backup in reversed(entries):
            if backup is None:
                target.unlink(missing_ok=True)
                self._fsync_directory(target.parent)
            else:
                self._restore_backup(backup, target)
        self._clear_transaction_manifest()
        for _, temporary, backup in entries:
            self._cleanup_transaction_file(temporary)
            if backup is not None:
                self._cleanup_transaction_file(backup)
        self._invalidate_indexes()
        logger.warning("storage_interrupted_transaction_recovered")

    def _atomic_write_many(self, writes: Sequence[tuple[Path, str]]) -> None:
        """Atomically replace a batch with startup rollback after hard crashes."""

        staged: list[tuple[Path, Path, Path | None]] = []
        manifest_written = False
        cleanup_staged = True
        try:
            if not writes:
                return
            for target, text in writes:
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                        handle.write(text)
                        handle.flush()
                        os.fsync(handle.fileno())
                except BaseException:
                    self._cleanup_transaction_file(temporary)
                    raise
                backup: Path | None = None
                if target.exists():
                    backup_descriptor, backup_name = tempfile.mkstemp(
                        prefix=f".{target.name}.", suffix=".backup.tmp", dir=target.parent
                    )
                    backup = Path(backup_name)
                    try:
                        with os.fdopen(backup_descriptor, "wb") as destination:
                            with target.open("rb") as source:
                                shutil.copyfileobj(source, destination)
                                destination.flush()
                                os.fsync(destination.fileno())
                    except BaseException:
                        self._cleanup_transaction_file(backup)
                        self._cleanup_transaction_file(temporary)
                        raise
                staged.append((target, temporary, backup))

            for directory in {target.parent for target, _, _ in staged}:
                self._fsync_directory(directory)
            self._write_transaction_manifest(staged)
            manifest_written = True
            for target, temporary, backup in staged:
                os.replace(temporary, target)
                self._fsync_directory(target.parent)
            self._clear_transaction_manifest()
            manifest_written = False
        except BaseException as original_error:
            rollback_errors: list[str] = []
            # Treat an installed on-disk manifest as authoritative even if an
            # interruption happened after its os.replace() but before the
            # in-memory flag assignment.  Restore the complete staged set:
            # os.replace() itself may have succeeded before raising, and
            # restoring an untouched target from its backup is idempotent.
            manifest_installed = (
                manifest_written or self.transaction_manifest_path.exists()
            )
            if manifest_installed:
                # From this point the manifest owns recovery.  Preserve every
                # backup if rollback itself is interrupted by another
                # BaseException; only re-enable cleanup after the manifest has
                # been cleared successfully.
                cleanup_staged = False
                for target, temporary, backup in reversed(staged):
                    try:
                        # A failed os.replace() that left its source temporary
                        # in place did not mutate this target.  Skipping it
                        # avoids needlessly invoking the same failing replace
                        # operation during rollback.  If the source vanished,
                        # replacement may have completed before raising and the
                        # original snapshot must be restored.
                        if temporary.exists():
                            continue
                        if backup is None:
                            target.unlink(missing_ok=True)
                            self._fsync_directory(target.parent)
                        else:
                            self._restore_backup(backup, target)
                    except Exception as rollback_error:
                        rollback_errors.append(f"{target}: {rollback_error}")
                if not rollback_errors:
                    try:
                        self._clear_transaction_manifest()
                        manifest_written = False
                        cleanup_staged = True
                    except OSError as rollback_error:
                        rollback_errors.append(
                            f"{self.transaction_manifest_path}: {rollback_error}"
                        )
            self._invalidate_indexes()
            if rollback_errors:
                cleanup_staged = False
                raise StorageError(
                    "storage batch failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from original_error
            raise
        finally:
            if cleanup_staged:
                for _, temporary, backup in staged:
                    self._cleanup_transaction_file(temporary)
                    if backup is not None:
                        self._cleanup_transaction_file(backup)

    def _validated_posts(self, posts: Sequence[XPost]) -> list[XPost]:
        if isinstance(posts, (str, bytes)) or not isinstance(posts, Sequence):
            raise StorageValidationError("posts must be a sequence of XPost values")
        result: list[XPost] = []
        for post in posts:
            clone = _clone_model(post, XPost)
            _validate_post_identity(clone)
            # Keep every validated sighting until memberships have been
            # collected.  Entity deduplication alone must not discard a second
            # List relation for the same post within one caller batch.
            result.append(clone)
        return result

    def _merge_entities(
        self,
        posts: Sequence[XPost],
        forced_list_id: str | None,
    ) -> tuple[
        dict[str, XPost],
        dict[str, XAuthor],
        set[tuple[str, str]],
        set[str],
    ]:
        stored_posts = self._read_model_jsonl(self.posts_path, XPost, "id")
        existing_post_ids = set(stored_posts)
        stored_authors = self._read_model_jsonl(self.authors_path, XAuthor, "id")
        memberships = self._read_memberships()
        self._validate_membership_targets(memberships, stored_posts)
        for post in posts:
            post_id = post.id
            # The first persisted post is the canonical global entity.  Later
            # sightings in other Lists add membership without creating a copy.
            stored_posts.setdefault(post_id, post)
            if post.author is not None:
                stored_authors[post.author.id] = _clone_model(post.author, XAuthor)
            list_id = forced_list_id or post.source_list_id
            memberships.add((post_id, _require_identifier(list_id, "list_id")))
        return (
            stored_posts,
            stored_authors,
            memberships,
            set(stored_posts).difference(existing_post_ids),
        )

    def _entity_writes(
        self,
        posts: Sequence[XPost],
        forced_list_id: str | None,
    ) -> tuple[
        list[tuple[Path, str]],
        dict[str, XPost],
        set[tuple[str, str]],
        set[str],
    ]:
        stored_posts, authors, memberships, inserted_ids = self._merge_entities(
            posts, forced_list_id
        )
        # All three files are staged before the first replace.  The checkpoint,
        # when present, is appended by commit_list_batch and therefore last.
        writes = [
            (self.posts_path, self._render_model_jsonl(stored_posts, XPost)),
            (self.authors_path, self._render_model_jsonl(authors, XAuthor)),
            (self.memberships_path, self._render_memberships(memberships)),
        ]
        return writes, stored_posts, memberships, inserted_ids

    def _refresh_indexes(
        self,
        posts: Mapping[str, XPost],
        memberships: set[tuple[str, str]],
    ) -> None:
        self._post_ids = set(posts)
        self._post_signature = self._signature(self.posts_path)
        self._membership_ids = set(memberships)
        self._membership_signature = self._signature(self.memberships_path)

    def has_post(self, post_id: str) -> bool:
        key = _require_identifier(post_id, "post_id")
        with self._guard():
            signature = self._signature(self.posts_path)
            if self._post_ids is None or signature != self._post_signature:
                posts = self._read_model_jsonl(self.posts_path, XPost, "id")
                self._post_ids = set(posts)
                self._post_signature = signature
            return key in self._post_ids

    def has_membership(self, post_id: str, list_id: str) -> bool:
        key = (
            _require_identifier(post_id, "post_id"),
            _require_identifier(list_id, "list_id"),
        )
        with self._guard():
            signature = self._signature(self.memberships_path)
            if self._membership_ids is None or signature != self._membership_signature:
                self._membership_ids = self._read_memberships()
                self._membership_signature = signature
            exists = key in self._membership_ids
            if exists and not self.has_post(key[0]):
                raise StorageCorruptionError(
                    f"membership {key!r} references a missing post"
                )
            return exists

    def save_posts(self, posts: Sequence[XPost]) -> None:
        validated = self._validated_posts(posts)
        with self._guard():
            writes, merged_posts, memberships, _ = self._entity_writes(validated, None)
            self._atomic_write_many(writes)
            self._refresh_indexes(merged_posts, memberships)

    def commit_list_batch(
        self,
        posts: Sequence[XPost],
        list_id: str,
        state: ListSyncState,
        *,
        expected_newest_post_id: str | None | object = _CHECKPOINT_UNSET,
        run: IngestionRun | None = None,
    ) -> set[str]:
        list_key = _require_identifier(list_id, "list_id")
        validated = self._validated_posts(posts)
        state_clone = _safe_sync_state(state)
        if state_clone.list_id != list_key:
            raise StorageValidationError("state.list_id must match list_id")
        expected = expected_newest_post_id
        if expected is not _CHECKPOINT_UNSET and expected is not None:
            expected = _require_identifier(expected, "expected_newest_post_id")
        run_clone = _safe_ingestion_run(run) if run is not None else None
        with self._guard():
            states = self._read_sync_states()
            current = states.get(list_key)
            current_id = current.newest_post_id if current is not None else None
            if expected is not _CHECKPOINT_UNSET and current_id != expected:
                raise StorageConflictError(
                    f"List {list_key!r} checkpoint changed concurrently"
                )
            writes, merged_posts, memberships, inserted_ids = self._entity_writes(
                validated, list_key
            )
            state_clone.last_status = "success" if inserted_ids else "no_new_posts"
            if run_clone is not None:
                run_path = self._run_path(run_clone.run_id)
                if not run_path.is_file():
                    raise StorageValidationError(
                        f"ingestion run {run_clone.run_id!r} does not exist"
                    )
                current_run = self._read_run_path(run_path)
                if current_run.ingestion_owner_id != run_clone.ingestion_owner_id:
                    raise StorageOwnershipConflictError(
                        f"ingestion run {run_clone.run_id!r} owner changed concurrently"
                    )
                _preserve_report_state(run_clone, current_run)
                inserted_in_order = [
                    post.id for post in validated if post.id in inserted_ids
                ]
                run_clone.new_post_ids = list(
                    dict.fromkeys([*run_clone.new_post_ids, *inserted_in_order])
                )
                run_clone.new_post_count = len(run_clone.new_post_ids)
                if list_key not in run_clone.successful_lists:
                    run_clone.successful_lists.append(list_key)
                # The pending report work item is durable before any entity or
                # checkpoint replacement.  A hard crash can therefore leave a
                # replayable run, never a checkpoint with no report ownership.
                writes.insert(0, (run_path, self._render_run(run_clone)))
            states[list_key] = state_clone
            # This final write is the ingestion transaction boundary.
            writes.append((self.sync_states_path, self._render_sync_states(states)))
            self._atomic_write_many(writes)
            self._refresh_indexes(merged_posts, memberships)
            return inserted_ids

    def get_sync_state(self, list_id: str) -> ListSyncState | None:
        key = _require_identifier(list_id, "list_id")
        with self._guard():
            state = self._read_sync_states().get(key)
            return None if state is None else _clone_model(state, ListSyncState)

    def save_sync_state(
        self,
        state: ListSyncState,
        *,
        expected_newest_post_id: str | None | object = _CHECKPOINT_UNSET,
    ) -> None:
        clone = _safe_sync_state(state)
        key = _require_identifier(clone.list_id, "list_id")
        expected = expected_newest_post_id
        if expected is not _CHECKPOINT_UNSET and expected is not None:
            expected = _require_identifier(expected, "expected_newest_post_id")
        with self._guard():
            states = self._read_sync_states()
            current = states.get(key)
            current_id = current.newest_post_id if current is not None else None
            if expected is not _CHECKPOINT_UNSET and current_id != expected:
                raise StorageConflictError(
                    f"List {key!r} checkpoint changed concurrently"
                )
            states[key] = clone
            self._atomic_write_many(
                [(self.sync_states_path, self._render_sync_states(states))]
            )

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"run_{_validate_run_id(run_id)}.json"

    def create_run(self, run: IngestionRun) -> None:
        clone = _safe_ingestion_run(run)
        path = self._run_path(clone.run_id)
        with self._guard():
            if path.exists():
                raise StorageValidationError(f"ingestion run {clone.run_id!r} already exists")
            self._atomic_write_many([(path, self._render_run(clone))])

    def update_run(
        self,
        run: IngestionRun,
        *,
        expected_report_status: str | object = _CHECKPOINT_UNSET,
        expected_report_claim_id: str | None | object = _CHECKPOINT_UNSET,
    ) -> None:
        clone = _safe_ingestion_run(run)
        path = self._run_path(clone.run_id)
        expected_status = expected_report_status
        if expected_status is not _CHECKPOINT_UNSET and not isinstance(
            expected_status, str
        ):
            raise StorageValidationError("expected_report_status must be a string")
        expected_claim_id = expected_report_claim_id
        if expected_claim_id is not _CHECKPOINT_UNSET and expected_claim_id is not None:
            expected_claim_id = _require_identifier(
                expected_claim_id, "expected_report_claim_id"
            )
        with self._guard():
            if not path.is_file():
                raise StorageValidationError(f"ingestion run {clone.run_id!r} does not exist")
            # Refuse to overwrite an already corrupted record silently.
            current = self._read_run_path(path)
            if (
                expected_status is not _CHECKPOINT_UNSET
                and current.report_status != expected_status
            ):
                raise StorageConflictError(
                    f"ingestion run {clone.run_id!r} report status changed concurrently"
                )
            if (
                expected_claim_id is not _CHECKPOINT_UNSET
                and current.report_claim_id != expected_claim_id
            ):
                raise StorageConflictError(
                    f"ingestion run {clone.run_id!r} report claim changed concurrently"
                )
            self._atomic_write_many([(path, self._render_run(clone))])

    def update_ingestion_run(
        self,
        run: IngestionRun,
        *,
        expected_ingestion_owner_id: str | None | object = _CHECKPOINT_UNSET,
        expected_ingestion_heartbeat_at: datetime | None | object = _CHECKPOINT_UNSET,
    ) -> None:
        """Merge ingestion progress while retaining any concurrent report claim."""

        clone = _safe_ingestion_run(run)
        path = self._run_path(clone.run_id)
        expected_owner = expected_ingestion_owner_id
        if expected_owner is not _CHECKPOINT_UNSET and expected_owner is not None:
            expected_owner = _require_identifier(
                expected_owner, "expected_ingestion_owner_id"
            )
        expected_heartbeat = expected_ingestion_heartbeat_at
        if expected_heartbeat is not _CHECKPOINT_UNSET and expected_heartbeat is not None:
            try:
                expected_heartbeat = ensure_aware_datetime(
                    expected_heartbeat,
                    field_name="expected_ingestion_heartbeat_at",
                )
            except (TypeError, ValueError) as exc:
                raise StorageValidationError(str(exc)) from exc
        with self._guard():
            if not path.is_file():
                raise StorageValidationError(f"ingestion run {clone.run_id!r} does not exist")
            current = self._read_run_path(path)
            if (
                expected_owner is not _CHECKPOINT_UNSET
                and current.ingestion_owner_id != expected_owner
            ):
                raise StorageOwnershipConflictError(
                    f"ingestion run {clone.run_id!r} owner changed concurrently"
                )
            if (
                expected_heartbeat is not _CHECKPOINT_UNSET
                and current.ingestion_heartbeat_at != expected_heartbeat
            ):
                raise StorageOwnershipConflictError(
                    f"ingestion run {clone.run_id!r} heartbeat changed concurrently"
                )
            _preserve_report_state(clone, current)
            self._atomic_write_many([(path, self._render_run(clone))])

    def get_run(self, run_id: str) -> IngestionRun | None:
        path = self._run_path(run_id)
        with self._guard():
            if not path.exists():
                return None
            return _clone_model(self._read_run_path(path), IngestionRun)

    # Explicit aliases make the interface self-documenting for callers that
    # distinguish ingestion runs from legacy HTML report history records.
    create_ingestion_run = create_run
    get_ingestion_run = get_run

    def read_posts(self, list_id: str | None = None) -> list[XPost]:
        with self._guard():
            posts = self._read_model_jsonl(self.posts_path, XPost, "id")
            self._post_ids = set(posts)
            self._post_signature = self._signature(self.posts_path)
            if list_id is None:
                selected = (posts[post_id] for post_id in sorted(posts))
            else:
                key = _require_identifier(list_id, "list_id")
                memberships = self._read_memberships()
                self._validate_membership_targets(memberships, posts)
                self._membership_ids = set(memberships)
                self._membership_signature = self._signature(self.memberships_path)
                member_ids = {post_id for post_id, source in memberships if source == key}
                missing = member_ids.difference(posts)
                if missing:
                    raise StorageCorruptionError(
                        f"memberships reference missing posts: {sorted(missing)!r}"
                    )
                selected = (posts[post_id] for post_id in sorted(member_ids))
            return [_clone_model(post, XPost) for post in selected]

    def read_authors(self) -> list[XAuthor]:
        with self._guard():
            authors = self._read_model_jsonl(self.authors_path, XAuthor, "id")
            return [_clone_model(authors[key], XAuthor) for key in sorted(authors)]

    def read_memberships(self) -> list[dict[str, str]]:
        with self._guard():
            values = self._read_memberships()
            posts = self._read_model_jsonl(self.posts_path, XPost, "id")
            self._validate_membership_targets(values, posts)
            self._membership_ids = set(values)
            self._membership_signature = self._signature(self.memberships_path)
            return [
                {"post_id": post_id, "list_id": list_id}
                for post_id, list_id in sorted(values, key=lambda item: (item[1], item[0]))
            ]

    def read_sync_states(self) -> list[ListSyncState]:
        with self._guard():
            values = self._read_sync_states()
            return [_clone_model(values[key], ListSyncState) for key in sorted(values)]

    def read_runs(self) -> list[IngestionRun]:
        with self._guard():
            runs: list[IngestionRun] = []
            for path in sorted(self.runs_dir.glob("run_*.json")):
                runs.append(_clone_model(self._read_run_path(path), IngestionRun))
            return runs

    def export_data(self) -> dict[str, Any]:
        """Return a validated, detached snapshot for migration or reporting."""
        with self._guard():
            return {
                "posts": [post.to_dict() for post in self.read_posts()],
                "authors": [author.to_dict() for author in self.read_authors()],
                "memberships": self.read_memberships(),
                "sync_states": [state.to_dict() for state in self.read_sync_states()],
                "runs": [run.to_dict() for run in self.read_runs()],
            }


class InMemoryPostStore:
    """Behavior-compatible, fully atomic store for offline tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._posts: dict[str, XPost] = {}
        self._authors: dict[str, XAuthor] = {}
        self._memberships: set[tuple[str, str]] = set()
        self._states: dict[str, ListSyncState] = {}
        self._runs: dict[str, IngestionRun] = {}

    def _save_validated(
        self,
        posts: Sequence[XPost],
        forced_list_id: str | None,
    ) -> None:
        values: list[XPost] = []
        for post in posts:
            clone = _clone_model(post, XPost)
            _validate_post_identity(clone)
            values.append(clone)
        # Validation is complete before any dictionary is changed.
        for post in values:
            post_id = post.id
            self._posts.setdefault(post_id, post)
            if post.author is not None:
                self._authors[post.author.id] = _clone_model(post.author, XAuthor)
            list_id = forced_list_id or post.source_list_id
            self._memberships.add(
                (post_id, _require_identifier(list_id, "list_id"))
            )

    def has_post(self, post_id: str) -> bool:
        key = _require_identifier(post_id, "post_id")
        with self._lock:
            return key in self._posts

    def has_membership(self, post_id: str, list_id: str) -> bool:
        key = (
            _require_identifier(post_id, "post_id"),
            _require_identifier(list_id, "list_id"),
        )
        with self._lock:
            return key in self._memberships

    def save_posts(self, posts: Sequence[XPost]) -> None:
        with self._lock:
            self._save_validated(posts, None)

    def commit_list_batch(
        self,
        posts: Sequence[XPost],
        list_id: str,
        state: ListSyncState,
        *,
        expected_newest_post_id: str | None | object = _CHECKPOINT_UNSET,
        run: IngestionRun | None = None,
    ) -> set[str]:
        list_key = _require_identifier(list_id, "list_id")
        state_clone = _safe_sync_state(state)
        if state_clone.list_id != list_key:
            raise StorageValidationError("state.list_id must match list_id")
        expected = expected_newest_post_id
        if expected is not _CHECKPOINT_UNSET and expected is not None:
            expected = _require_identifier(expected, "expected_newest_post_id")
        run_clone = _safe_ingestion_run(run) if run is not None else None
        # Validate every post into detached objects before the atomic mutation.
        validated = [_clone_model(post, XPost) for post in posts]
        with self._lock:
            current = self._states.get(list_key)
            current_id = current.newest_post_id if current is not None else None
            if expected is not _CHECKPOINT_UNSET and current_id != expected:
                raise StorageConflictError(
                    f"List {list_key!r} checkpoint changed concurrently"
                )
            if run_clone is not None and run_clone.run_id not in self._runs:
                raise StorageValidationError(
                    f"ingestion run {run_clone.run_id!r} does not exist"
                )
            inserted_ids = {post.id for post in validated if post.id not in self._posts}
            state_clone.last_status = "success" if inserted_ids else "no_new_posts"
            if run_clone is not None:
                current_run = self._runs[run_clone.run_id]
                if current_run.ingestion_owner_id != run_clone.ingestion_owner_id:
                    raise StorageOwnershipConflictError(
                        f"ingestion run {run_clone.run_id!r} owner changed concurrently"
                    )
                _preserve_report_state(run_clone, current_run)
                inserted_in_order = [
                    post.id for post in validated if post.id in inserted_ids
                ]
                run_clone.new_post_ids = list(
                    dict.fromkeys([*run_clone.new_post_ids, *inserted_in_order])
                )
                run_clone.new_post_count = len(run_clone.new_post_ids)
                if list_key not in run_clone.successful_lists:
                    run_clone.successful_lists.append(list_key)
            self._save_validated(validated, list_key)
            if run_clone is not None:
                self._runs[run_clone.run_id] = run_clone
            self._states[list_key] = state_clone
            return inserted_ids

    def get_sync_state(self, list_id: str) -> ListSyncState | None:
        key = _require_identifier(list_id, "list_id")
        with self._lock:
            value = self._states.get(key)
            return None if value is None else _clone_model(value, ListSyncState)

    def save_sync_state(
        self,
        state: ListSyncState,
        *,
        expected_newest_post_id: str | None | object = _CHECKPOINT_UNSET,
    ) -> None:
        clone = _safe_sync_state(state)
        expected = expected_newest_post_id
        if expected is not _CHECKPOINT_UNSET and expected is not None:
            expected = _require_identifier(expected, "expected_newest_post_id")
        with self._lock:
            key = _require_identifier(clone.list_id, "list_id")
            current = self._states.get(key)
            current_id = current.newest_post_id if current is not None else None
            if expected is not _CHECKPOINT_UNSET and current_id != expected:
                raise StorageConflictError(
                    f"List {key!r} checkpoint changed concurrently"
                )
            self._states[key] = clone

    def create_run(self, run: IngestionRun) -> None:
        clone = _safe_ingestion_run(run)
        key = _validate_run_id(clone.run_id)
        with self._lock:
            if key in self._runs:
                raise StorageValidationError(f"ingestion run {key!r} already exists")
            self._runs[key] = clone

    def update_run(
        self,
        run: IngestionRun,
        *,
        expected_report_status: str | object = _CHECKPOINT_UNSET,
        expected_report_claim_id: str | None | object = _CHECKPOINT_UNSET,
    ) -> None:
        clone = _safe_ingestion_run(run)
        key = _validate_run_id(clone.run_id)
        expected_status = expected_report_status
        if expected_status is not _CHECKPOINT_UNSET and not isinstance(
            expected_status, str
        ):
            raise StorageValidationError("expected_report_status must be a string")
        expected_claim_id = expected_report_claim_id
        if expected_claim_id is not _CHECKPOINT_UNSET and expected_claim_id is not None:
            expected_claim_id = _require_identifier(
                expected_claim_id, "expected_report_claim_id"
            )
        with self._lock:
            if key not in self._runs:
                raise StorageValidationError(f"ingestion run {key!r} does not exist")
            if (
                expected_status is not _CHECKPOINT_UNSET
                and self._runs[key].report_status != expected_status
            ):
                raise StorageConflictError(
                    f"ingestion run {key!r} report status changed concurrently"
                )
            if (
                expected_claim_id is not _CHECKPOINT_UNSET
                and self._runs[key].report_claim_id != expected_claim_id
            ):
                raise StorageConflictError(
                    f"ingestion run {key!r} report claim changed concurrently"
                )
            self._runs[key] = clone

    def update_ingestion_run(
        self,
        run: IngestionRun,
        *,
        expected_ingestion_owner_id: str | None | object = _CHECKPOINT_UNSET,
        expected_ingestion_heartbeat_at: datetime | None | object = _CHECKPOINT_UNSET,
    ) -> None:
        """Merge ingestion progress while retaining any concurrent report claim."""

        clone = _safe_ingestion_run(run)
        key = _validate_run_id(clone.run_id)
        expected_owner = expected_ingestion_owner_id
        if expected_owner is not _CHECKPOINT_UNSET and expected_owner is not None:
            expected_owner = _require_identifier(
                expected_owner, "expected_ingestion_owner_id"
            )
        expected_heartbeat = expected_ingestion_heartbeat_at
        if expected_heartbeat is not _CHECKPOINT_UNSET and expected_heartbeat is not None:
            try:
                expected_heartbeat = ensure_aware_datetime(
                    expected_heartbeat,
                    field_name="expected_ingestion_heartbeat_at",
                )
            except (TypeError, ValueError) as exc:
                raise StorageValidationError(str(exc)) from exc
        with self._lock:
            if key not in self._runs:
                raise StorageValidationError(f"ingestion run {key!r} does not exist")
            if (
                expected_owner is not _CHECKPOINT_UNSET
                and self._runs[key].ingestion_owner_id != expected_owner
            ):
                raise StorageOwnershipConflictError(
                    f"ingestion run {key!r} owner changed concurrently"
                )
            if (
                expected_heartbeat is not _CHECKPOINT_UNSET
                and self._runs[key].ingestion_heartbeat_at != expected_heartbeat
            ):
                raise StorageOwnershipConflictError(
                    f"ingestion run {key!r} heartbeat changed concurrently"
                )
            _preserve_report_state(clone, self._runs[key])
            self._runs[key] = clone

    def get_run(self, run_id: str) -> IngestionRun | None:
        key = _validate_run_id(run_id)
        with self._lock:
            value = self._runs.get(key)
            return None if value is None else _clone_model(value, IngestionRun)

    create_ingestion_run = create_run
    get_ingestion_run = get_run

    def read_posts(self, list_id: str | None = None) -> list[XPost]:
        with self._lock:
            if list_id is None:
                values = (self._posts[post_id] for post_id in sorted(self._posts))
            else:
                key = _require_identifier(list_id, "list_id")
                post_ids = {
                    post_id for post_id, source in self._memberships if source == key
                }
                values = (self._posts[post_id] for post_id in sorted(post_ids))
            return [_clone_model(post, XPost) for post in values]

    def read_authors(self) -> list[XAuthor]:
        with self._lock:
            return [
                _clone_model(self._authors[key], XAuthor) for key in sorted(self._authors)
            ]

    def read_memberships(self) -> list[dict[str, str]]:
        with self._lock:
            return [
                {"post_id": post_id, "list_id": list_id}
                for post_id, list_id in sorted(
                    self._memberships, key=lambda item: (item[1], item[0])
                )
            ]

    def read_sync_states(self) -> list[ListSyncState]:
        with self._lock:
            return [
                _clone_model(self._states[key], ListSyncState)
                for key in sorted(self._states)
            ]

    def read_runs(self) -> list[IngestionRun]:
        with self._lock:
            return [
                _clone_model(self._runs[key], IngestionRun) for key in sorted(self._runs)
            ]

    def export_data(self) -> dict[str, Any]:
        with self._lock:
            return {
                "posts": [post.to_dict() for post in self.read_posts()],
                "authors": [author.to_dict() for author in self.read_authors()],
                "memberships": self.read_memberships(),
                "sync_states": [state.to_dict() for state in self.read_sync_states()],
                "runs": [run.to_dict() for run in self.read_runs()],
            }


__all__ = [
    "FilePostStore",
    "InMemoryPostStore",
    "PostStore",
    "StorageConflictError",
    "StorageCorruptionError",
    "StorageError",
    "StorageOwnershipConflictError",
    "StorageValidationError",
]
