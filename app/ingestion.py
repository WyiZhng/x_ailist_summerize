"""Reliable incremental ingestion independent from the HTTP dashboard.

The X List Posts endpoint does not currently expose ``since_id``.  This
coordinator therefore walks newest-first pages until it reaches (or passes) a
per-List Snowflake checkpoint.  A checkpoint is committed only after a
continuous result has been normalized and durably stored.  Summary/report
failures happen after this boundary and never cause an already stored post to
be fetched again.
"""

from __future__ import annotations

import copy
import inspect
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Literal, Protocol, Sequence, runtime_checkable

from .models import IngestionRun, ListSyncState, XPost, utc_now
from .security import redact_sensitive_text
from .storage import (
    PostStore,
    StorageConflictError,
    StorageOwnershipConflictError,
)
from .x_api_provider import XFetchPage, compare_post_ids


logger = logging.getLogger(__name__)

ListIngestionStatus = Literal["success", "no_new_posts", "incomplete", "failed"]
IngestionStatus = Literal[
    "success", "no_new_posts", "partial_success", "failed"
]
IngestionProgressCallback = Callable[[dict[str, Any]], Any]
INGESTION_RUN_LEASE = timedelta(hours=1)


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.astimezone()
    return value


def ingestion_run_is_stale(
    run: IngestionRun,
    *,
    now: datetime | None = None,
    lease: timedelta = INGESTION_RUN_LEASE,
) -> bool:
    """Return whether a running ingestion owner's durable lease has expired."""

    if run.status != "running":
        return False
    heartbeat = run.ingestion_heartbeat_at or run.started_at
    current = _aware_datetime(now or utc_now())
    return current >= _aware_datetime(heartbeat) + lease


@runtime_checkable
class XIngestionProvider(Protocol):
    """Page-level source used by :class:`IncrementalIngestionService`."""

    async def fetch_page(
        self,
        list_id: str,
        *,
        pagination_token: str | None = None,
        max_results: int = 100,
        page_number: int = 1,
        run_id: str | None = None,
    ) -> XFetchPage:
        """Return one newest-first page of canonical posts."""


@dataclass(slots=True)
class ListIngestionResult:
    """Outcome for one independently committed X List."""

    list_id: str
    status: ListIngestionStatus
    fetched_count: int = 0
    new_post_count: int = 0
    duplicate_count: int = 0
    page_count: int = 0
    stop_reason: str = ""
    pagination_complete: bool = False
    checkpoint_post_id: str | None = None
    new_posts: list[XPost] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status in {"success", "no_new_posts"}

    def to_dict(self, *, include_posts: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "list_id": self.list_id,
            "status": self.status,
            "success": self.success,
            "fetched_count": self.fetched_count,
            "new_post_count": self.new_post_count,
            "duplicate_count": self.duplicate_count,
            "page_count": self.page_count,
            "stop_reason": self.stop_reason,
            "pagination_complete": self.pagination_complete,
            "checkpoint_post_id": self.checkpoint_post_id,
            "error": self.error,
        }
        if include_posts:
            value["new_posts"] = [post.to_dict() for post in self.new_posts]
        else:
            value["new_post_ids"] = [post.id for post in self.new_posts]
        return value


@dataclass(slots=True)
class IngestionResult:
    """Outcome for a multi-List ingestion run."""

    run: IngestionRun
    list_results: list[ListIngestionResult] = field(default_factory=list)
    new_posts: list[XPost] = field(default_factory=list)

    @property
    def status(self) -> IngestionStatus:
        return self.run.status  # type: ignore[return-value]

    @property
    def success(self) -> bool:
        return self.status in {"success", "no_new_posts", "partial_success"}

    @property
    def partial(self) -> bool:
        return self.status == "partial_success"

    def to_dict(self, *, include_posts: bool = False) -> dict[str, Any]:
        value = self.run.to_dict()
        value["list_results"] = [
            item.to_dict(include_posts=include_posts) for item in self.list_results
        ]
        if include_posts:
            value["new_posts"] = [post.to_dict() for post in self.new_posts]
        return value


class IncompleteIngestionError(RuntimeError):
    """Raised when pagination cannot prove a gap-free checkpoint boundary."""

    def __init__(self, message: str, *, stop_reason: str) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason


def normalize_list_id(value: str) -> str:
    """Extract a numeric List ID while retaining explicit mock identifiers."""

    text = str(value).strip()
    if not text:
        raise ValueError("List ID must not be empty")
    if text.startswith("mock://"):
        return text
    if re.fullmatch(r"[0-9]{1,19}", text):
        return text
    match = re.fullmatch(
        r"(?:https?://)?(?:www\.)?(?:x|twitter)\.com/"
        r"(?:i/)?lists/([0-9]{1,19})(?:[/?#].*)?",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else text


def _clone_state(state: ListSyncState | None, list_id: str) -> ListSyncState:
    if state is None:
        return ListSyncState(list_id=list_id)
    return ListSyncState.from_dict(state.to_dict())


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class IncrementalIngestionService:
    """Fetch, normalize, deduplicate and durably checkpoint X List posts."""

    def __init__(
        self,
        *,
        provider: XIngestionProvider,
        store: PostStore,
        page_size: int = 100,
        max_pages: int = 20,
        initial_fetch_limit: int = 100,
        now_factory: Callable[[], datetime] = utc_now,
        owner_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        progress_callback: IngestionProgressCallback | None = None,
        sensitive_values: Sequence[str] = (),
    ) -> None:
        if not callable(getattr(provider, "fetch_page", None)):
            raise TypeError("provider must implement fetch_page()")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 100:
            raise ValueError("page_size must be an integer between 1 and 100")
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
            raise ValueError("max_pages must be a positive integer")
        if (
            isinstance(initial_fetch_limit, bool)
            or not isinstance(initial_fetch_limit, int)
            or initial_fetch_limit < 1
        ):
            raise ValueError("initial_fetch_limit must be a positive integer")
        self.provider = provider
        self.store = store
        self.page_size = page_size
        self.max_pages = max_pages
        self.initial_fetch_limit = initial_fetch_limit
        self._now = now_factory
        self._owner_id_factory = owner_id_factory
        self._progress_callback = progress_callback
        self._sensitive_values = tuple(item for item in sensitive_values if item)

    def _safe_error(self, error: Any) -> str:
        return redact_sensitive_text(str(error), secrets=self._sensitive_values)

    async def _emit(self, **event: Any) -> None:
        if self._progress_callback is None:
            return
        try:
            await _maybe_await(self._progress_callback(dict(event)))
        except Exception:
            logger.warning("ingestion_progress_callback_failed")

    def _is_checkpoint_boundary(
        self,
        post_id: str,
        list_id: str,
        prior_state: ListSyncState | None,
    ) -> bool:
        if prior_state and prior_state.newest_post_id:
            ordering = compare_post_ids(post_id, prior_state.newest_post_id)
            if ordering in {-1, 0}:
                return True
            # If IDs cannot be ordered, exact per-List membership is the safe
            # fallback.  A global post hit alone must never stop another List.
            if ordering is None and self.store.has_membership(post_id, list_id):
                return True
            return False
        return self.store.has_membership(post_id, list_id)

    def _failure_state(
        self,
        prior_state: ListSyncState | None,
        *,
        list_id: str,
        run_id: str,
        attempted_at: datetime,
        error: str,
    ) -> ListSyncState:
        state = _clone_state(prior_state, list_id)
        state.last_attempt_at = attempted_at
        state.last_run_id = run_id
        state.last_status = "failed"
        state.last_error = error
        return state

    async def _ingest_list(
        self,
        list_id: str,
        *,
        run_record: IngestionRun,
        run_id: str,
        max_tweets: int,
        _conflict_attempt: int = 0,
    ) -> ListIngestionResult:
        attempted_at = self._now()
        prior_state = self.store.get_sync_state(list_id)
        first_sync = prior_state is None or prior_state.newest_post_id is None
        limit = self.initial_fetch_limit if first_sync else max_tweets
        token: str | None = None
        seen_tokens: set[str] = set()
        batch_ids: set[str] = set()
        candidate_posts: list[XPost] = []
        new_entity_posts: list[XPost] = []
        fetched_count = 0
        duplicate_count = 0
        page_count = 0
        stop_reason = ""
        pagination_complete = False
        newest_observed: XPost | None = None

        try:
            while page_count < self.max_pages:
                remaining = limit - fetched_count
                if remaining <= 0:
                    if first_sync:
                        stop_reason = "initial_limit"
                        pagination_complete = False
                        break
                    raise IncompleteIngestionError(
                        "Incremental window ended before the previous checkpoint",
                        stop_reason="max_tweets_before_checkpoint",
                    )

                page_number = page_count + 1
                page = await self.provider.fetch_page(
                    list_id,
                    pagination_token=token,
                    max_results=min(self.page_size, remaining),
                    page_number=page_number,
                    run_id=run_id,
                )
                if not isinstance(page, XFetchPage):
                    raise TypeError("provider.fetch_page() must return XFetchPage")
                page_count += 1
                fetched_count += len(page.posts)
                rate_remaining = page.rate_limit.remaining if page.rate_limit else None
                if page.raw_errors:
                    raise IncompleteIngestionError(
                        "X API returned data with partial errors",
                        stop_reason="partial_api_response",
                    )
                run_record.ingestion_heartbeat_at = self._now()
                self.store.update_ingestion_run(
                    run_record,
                    expected_ingestion_owner_id=run_record.ingestion_owner_id,
                )

                reached_checkpoint = False
                for post in page.posts:
                    if not isinstance(post, XPost):
                        raise TypeError("X API page contains a non-XPost value")
                    if not post.id:
                        raise IncompleteIngestionError(
                            "X API returned a post without an ID",
                            stop_reason="invalid_post_id",
                        )
                    post.source_list_id = list_id
                    if self._is_checkpoint_boundary(post.id, list_id, prior_state):
                        # Recovery for a prior entity-write success followed by
                        # checkpoint replacement failure: the per-List
                        # membership proves this post was durably normalized.
                        if first_sync and newest_observed is None:
                            newest_observed = post
                        reached_checkpoint = True
                        break
                    if newest_observed is None:
                        newest_observed = post
                    if post.id in batch_ids:
                        duplicate_count += 1
                        continue
                    batch_ids.add(post.id)
                    candidate_posts.append(post)
                    if self.store.has_post(post.id):
                        duplicate_count += 1
                    else:
                        new_entity_posts.append(post)

                logger.info(
                    "ingestion_page run_id=%s list_id=%s page_number=%s fetched_count=%s "
                    "new_count=%s duplicate_count=%s rate_limit_remaining=%s status=received",
                    run_id,
                    list_id,
                    page_count,
                    len(page.posts),
                    len(new_entity_posts),
                    duplicate_count,
                    rate_remaining,
                )
                await self._emit(
                    phase="ingesting",
                    run_id=run_id,
                    list_id=list_id,
                    page_number=page_count,
                    fetched_count=fetched_count,
                    new_count=len(new_entity_posts),
                    duplicate_count=duplicate_count,
                    rate_limit_remaining=rate_remaining,
                )

                if reached_checkpoint:
                    stop_reason = "checkpoint_reached"
                    pagination_complete = True
                    break
                if not page.next_token:
                    if prior_state and prior_state.newest_post_id and fetched_count:
                        raise IncompleteIngestionError(
                            "X API timeline ended before the previous checkpoint",
                            stop_reason="timeline_ended_before_checkpoint",
                        )
                    stop_reason = "empty" if fetched_count == 0 else "end_of_timeline"
                    pagination_complete = True
                    break
                if first_sync and fetched_count >= limit:
                    stop_reason = "initial_limit"
                    pagination_complete = False
                    break
                if page.next_token in seen_tokens:
                    raise IncompleteIngestionError(
                        "X API repeated a pagination token",
                        stop_reason="repeated_pagination_token",
                    )
                seen_tokens.add(page.next_token)
                token = page.next_token
            else:
                if first_sync:
                    stop_reason = "initial_page_limit"
                    pagination_complete = False
                else:
                    raise IncompleteIngestionError(
                        "Maximum page count reached before the previous checkpoint",
                        stop_reason="max_pages_before_checkpoint",
                    )

            state = _clone_state(prior_state, list_id)
            if newest_observed is not None:
                state.newest_post_id = newest_observed.id
                state.newest_post_created_at = newest_observed.created_at
            state.last_attempt_at = attempted_at
            state.last_success_at = self._now()
            state.last_run_id = run_id
            state.last_status = "success" if new_entity_posts else "no_new_posts"
            state.last_error = None
            estimated_new_ids = {post.id for post in new_entity_posts}
            commit_run = IngestionRun.from_dict(run_record.to_dict())
            commit_run.fetched_count += fetched_count
            commit_run.duplicate_count += duplicate_count
            try:
                inserted_ids = self.store.commit_list_batch(
                    candidate_posts,
                    list_id,
                    state,
                    expected_newest_post_id=(
                        prior_state.newest_post_id if prior_state is not None else None
                    ),
                    run=commit_run,
                )
            except StorageOwnershipConflictError:
                raise
            except StorageConflictError:
                if _conflict_attempt >= 2:
                    raise IncompleteIngestionError(
                        "List checkpoint kept changing during collection",
                        stop_reason="concurrent_checkpoint_conflict",
                    ) from None
                logger.warning(
                    "ingestion_checkpoint_conflict run_id=%s list_id=%s retry_attempt=%s",
                    run_id,
                    list_id,
                    _conflict_attempt + 1,
                )
                retried = await self._ingest_list(
                    list_id,
                    run_record=run_record,
                    run_id=run_id,
                    max_tweets=max_tweets,
                    _conflict_attempt=_conflict_attempt + 1,
                )
                retried.fetched_count += fetched_count
                retried.duplicate_count += duplicate_count
                retried.page_count += page_count
                return retried
            duplicate_count += len(estimated_new_ids.difference(inserted_ids))
            new_entity_posts = [
                post for post in candidate_posts if post.id in inserted_ids
            ]
            run_record.new_post_ids = list(
                dict.fromkeys(
                    [
                        *run_record.new_post_ids,
                        *(post.id for post in new_entity_posts),
                    ]
                )
            )
            run_record.new_post_count = len(run_record.new_post_ids)

            status: ListIngestionStatus = "success" if new_entity_posts else "no_new_posts"
            logger.info(
                "ingestion_list_complete run_id=%s list_id=%s page_number=%s "
                "fetched_count=%s new_count=%s duplicate_count=%s stop_reason=%s status=%s",
                run_id,
                list_id,
                page_count,
                fetched_count,
                len(new_entity_posts),
                duplicate_count,
                stop_reason,
                status,
            )
            return ListIngestionResult(
                list_id=list_id,
                status=status,
                fetched_count=fetched_count,
                new_post_count=len(new_entity_posts),
                duplicate_count=duplicate_count,
                page_count=page_count,
                stop_reason=stop_reason,
                pagination_complete=pagination_complete,
                checkpoint_post_id=state.newest_post_id,
                new_posts=[XPost.from_dict(post.to_dict()) for post in new_entity_posts],
            )
        except StorageOwnershipConflictError:
            raise
        except Exception as exc:
            safe_error = self._safe_error(exc)
            reason = getattr(exc, "stop_reason", type(exc).__name__)
            failure_state = self._failure_state(
                prior_state,
                list_id=list_id,
                run_id=run_id,
                attempted_at=attempted_at,
                error=safe_error,
            )
            try:
                self.store.save_sync_state(
                    failure_state,
                    expected_newest_post_id=(
                        prior_state.newest_post_id if prior_state is not None else None
                    ),
                )
            except StorageConflictError:
                logger.info(
                    "ingestion_failure_state_superseded run_id=%s list_id=%s status=failed",
                    run_id,
                    list_id,
                )
            except Exception:
                logger.error(
                    "ingestion_failure_state_write_failed run_id=%s list_id=%s status=failed",
                    run_id,
                    list_id,
                )
            logger.error(
                "ingestion_list_failed run_id=%s list_id=%s page_number=%s "
                "fetched_count=%s new_count=%s duplicate_count=%s stop_reason=%s status=failed",
                run_id,
                list_id,
                page_count,
                fetched_count,
                len(new_entity_posts),
                duplicate_count,
                reason,
            )
            return ListIngestionResult(
                list_id=list_id,
                status="incomplete" if isinstance(exc, IncompleteIngestionError) else "failed",
                fetched_count=fetched_count,
                new_post_count=0,
                duplicate_count=duplicate_count,
                page_count=page_count,
                stop_reason=str(reason),
                pagination_complete=False,
                checkpoint_post_id=prior_state.newest_post_id if prior_state else None,
                error=safe_error,
            )

    async def ingest(
        self,
        list_urls: Sequence[str],
        *,
        run_id: str,
        max_tweets: int = 100,
    ) -> IngestionResult:
        """Ingest Lists independently and persist a separate run record."""

        if isinstance(max_tweets, bool) or not isinstance(max_tweets, int) or max_tweets < 1:
            raise ValueError("max_tweets must be a positive integer")
        list_ids = [normalize_list_id(value) for value in list_urls]
        if not list_ids:
            raise ValueError("At least one List is required")
        started_at = self._now()
        owner_id = str(self._owner_id_factory()).strip()
        if not owner_id:
            raise ValueError("owner_id_factory must return a non-empty identifier")
        run = IngestionRun(
            run_id=run_id,
            started_at=started_at,
            status="running",
            ingestion_owner_id=owner_id,
            ingestion_heartbeat_at=started_at,
            requested_lists=list_ids,
        )
        self.store.create_run(run)
        results: list[ListIngestionResult] = []
        new_posts_by_id: dict[str, XPost] = {}

        try:
            for list_id in list_ids:
                result = await self._ingest_list(
                    list_id,
                    run_record=run,
                    run_id=run_id,
                    max_tweets=max_tweets,
                )
                results.append(result)
                run.fetched_count += result.fetched_count
                run.duplicate_count += result.duplicate_count
                if result.success:
                    if list_id not in run.successful_lists:
                        run.successful_lists.append(list_id)
                else:
                    if list_id not in run.failed_lists:
                        run.failed_lists.append(list_id)
                    run.errors.append(
                        {
                            "list_id": list_id,
                            "kind": result.stop_reason or "ingestion_failed",
                            "message": result.error or "List ingestion failed",
                        }
                    )
                for post in result.new_posts:
                    if post.id in new_posts_by_id:
                        run.duplicate_count += 1
                    else:
                        new_posts_by_id[post.id] = XPost.from_dict(post.to_dict())

            run.new_post_ids = list(new_posts_by_id)
            run.new_post_count = len(new_posts_by_id)
            if run.failed_lists and not run.successful_lists:
                run.status = "failed"
            elif run.failed_lists:
                run.status = "partial_success"
            elif not new_posts_by_id:
                run.status = "no_new_posts"
            else:
                run.status = "success"
            run.finished_at = self._now()
            run.ingestion_heartbeat_at = run.finished_at
            run.ingestion_owner_id = None
            self.store.update_ingestion_run(
                run,
                expected_ingestion_owner_id=owner_id,
            )
            logger.info(
                "ingestion_run_complete run_id=%s fetched_count=%s new_count=%s "
                "duplicate_count=%s status=%s",
                run_id,
                run.fetched_count,
                run.new_post_count,
                run.duplicate_count,
                run.status,
            )
            return IngestionResult(
                run=IngestionRun.from_dict(run.to_dict()),
                list_results=copy.deepcopy(results),
                new_posts=list(new_posts_by_id.values()),
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            safe_error = self._safe_error(exc)
            run.status = "failed"
            run.finished_at = self._now()
            run.ingestion_heartbeat_at = run.finished_at
            run.ingestion_owner_id = None
            run.errors.append({"kind": type(exc).__name__, "message": safe_error})
            try:
                self.store.update_ingestion_run(
                    run,
                    expected_ingestion_owner_id=owner_id,
                )
            except Exception:
                logger.error("ingestion_run_failure_record_failed run_id=%s", run_id)
            raise


__all__ = [
    "INGESTION_RUN_LEASE",
    "IncompleteIngestionError",
    "IncrementalIngestionService",
    "IngestionResult",
    "ListIngestionResult",
    "XIngestionProvider",
    "ingestion_run_is_stale",
    "normalize_list_id",
]
