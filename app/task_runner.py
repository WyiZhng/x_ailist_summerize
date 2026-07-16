"""Injectable digest task orchestration.

This module extracts the fetch/aggregate/summarize/report/history workflow from
the HTTP handler without requiring the existing UI to change immediately.  It
accepts the legacy fetcher and ``LLMProvider`` objects directly, but also offers
provider/factory seams for deterministic tests and future task backends.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import re
import threading
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .providers import (
    AggregatedDigest,
    MockReportGenerator,
    MockSummaryProvider,
    MockXIngestionProvider,
    MockXProvider,
    ReportGenerator,
    SummaryProvider,
    Tweet,
    XProvider,
)
from .config import atomic_write_json
from .ingestion import (
    IncrementalIngestionService,
    IngestionResult,
    ingestion_run_is_stale,
)
from .models import IngestionRun, XPost, utc_now
from .security import redact_sensitive_text
from .storage import (
    FilePostStore,
    PostStore,
    StorageConflictError,
    StorageOwnershipConflictError,
)


logger = logging.getLogger(__name__)


TaskStatus = Literal["succeeded", "partial", "failed", "no_new_posts"]
ProgressCallback = Callable[["TaskProgress"], Any]
ProviderFactory = Callable[..., Any]
Aggregator = Callable[[list[Tweet]], AggregatedDigest]
FetchDelayFactory = Callable[[int], float]
HISTORY_RESERVED_KEYS = frozenset(
    {
        "run_id",
        "name",
        "username",
        "tweets",
        "links",
        "profile_img",
        "members",
        "status",
        "completed_lists",
        "failed_lists",
        "generated_at",
    }
)
REPORT_CLAIM_LEASE = timedelta(hours=1)
REPORT_CLAIM_HEARTBEAT_SECONDS = 60.0


def _aware_datetime(value: datetime) -> datetime:
    """Normalize injected clocks without silently persisting a naive value."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.astimezone()
    return value


def report_claim_is_stale(
    run: IngestionRun,
    *,
    now: datetime | None = None,
    lease: timedelta = REPORT_CLAIM_LEASE,
) -> bool:
    """Return whether a durable report claim can be safely reclaimed."""

    if run.report_status != "generating":
        return False
    if run.report_claimed_at is None:
        # Legacy/interrupted claims have no owner lease and cannot make
        # progress; treating them as stale is the only recoverable behavior.
        return True
    current = _aware_datetime(now or utc_now())
    claimed = _aware_datetime(run.report_claimed_at)
    return current >= claimed + lease


@dataclass(frozen=True)
class DigestTaskRequest:
    """Inputs for one digest run."""

    list_urls: tuple[str, ...]
    max_tweets: int = 100
    output_dir: Path = Path("output")
    ai_model: str = ""
    report_filename: str | None = None
    history_metadata: Mapping[str, Any] = field(default_factory=dict)
    incremental: bool = False
    fetch_only: bool = False
    data_dir: Path = Path("data")
    page_size: int = 100
    max_pages: int = 20
    initial_fetch_limit: int = 100
    resume_ingestion_run_id: str | None = None

    def __post_init__(self) -> None:
        urls = tuple(str(url).strip() for url in self.list_urls if str(url).strip())
        if not urls:
            raise ValueError("At least one list URL or ID is required")
        if (
            not isinstance(self.max_tweets, int)
            or isinstance(self.max_tweets, bool)
            or self.max_tweets <= 0
        ):
            raise ValueError("max_tweets must be a positive integer")
        if self.report_filename:
            filename = Path(self.report_filename)
            if (
                filename.name != self.report_filename
                or not re.fullmatch(r"[A-Za-z0-9_.-]+\.html", self.report_filename)
            ):
                raise ValueError("report_filename must be a plain .html filename")
        metadata = dict(self.history_metadata)
        reserved = HISTORY_RESERVED_KEYS.intersection(metadata)
        if reserved:
            raise ValueError(
                "history_metadata cannot override reserved fields: "
                + ", ".join(sorted(reserved))
            )
        try:
            json.dumps(metadata, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("history_metadata must be JSON-serializable") from exc
        if self.fetch_only and self.resume_ingestion_run_id:
            raise ValueError("fetch_only cannot resume a report run")
        if self.fetch_only:
            object.__setattr__(self, "incremental", True)
        for name, value, maximum in (
            ("page_size", self.page_size, 100),
            ("max_pages", self.max_pages, None),
            ("initial_fetch_limit", self.initial_fetch_limit, None),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if maximum is not None and value > maximum:
                raise ValueError(f"{name} must be at most {maximum}")

        object.__setattr__(self, "list_urls", urls)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "history_metadata", metadata)

    @classmethod
    def from_values(
        cls,
        list_urls: Sequence[str],
        *,
        max_tweets: int = 100,
        output_dir: str | Path = "output",
        ai_model: str = "",
        report_filename: str | None = None,
        history_metadata: Mapping[str, Any] | None = None,
        incremental: bool = False,
        fetch_only: bool = False,
        data_dir: str | Path = "data",
        page_size: int = 100,
        max_pages: int = 20,
        initial_fetch_limit: int = 100,
        resume_ingestion_run_id: str | None = None,
    ) -> "DigestTaskRequest":
        return cls(
            list_urls=tuple(list_urls),
            max_tweets=max_tweets,
            output_dir=Path(output_dir),
            ai_model=ai_model,
            report_filename=report_filename,
            history_metadata=history_metadata or {},
            incremental=incremental,
            fetch_only=fetch_only,
            data_dir=Path(data_dir),
            page_size=page_size,
            max_pages=max_pages,
            initial_fetch_limit=initial_fetch_limit,
            resume_ingestion_run_id=resume_ingestion_run_id,
        )


@dataclass(frozen=True)
class TaskProgress:
    """One immutable task progress event."""

    phase: str
    percent: int
    message: str
    run_id: str = ""
    completed_lists: int = 0
    total_lists: int = 0
    failed_lists: int = 0
    current_list: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["timestamp"] = self.timestamp.isoformat()
        return value


@dataclass(frozen=True)
class ListFetchFailure:
    """Structured failure for an individual list."""

    list_url: str
    error: str


@dataclass
class DigestTaskResult:
    """Structured result returned for success, partial success, or failure."""

    status: TaskStatus
    started_at: datetime
    finished_at: datetime
    list_urls: tuple[str, ...]
    run_id: str = ""
    completed_lists: tuple[str, ...] = ()
    list_failures: tuple[ListFetchFailure, ...] = ()
    tweet_count: int = 0
    link_count: int = 0
    tweets: list[Tweet] = field(default_factory=list)
    aggregated: AggregatedDigest = field(default_factory=dict)
    summary: str = ""
    report_path: Path | None = None
    history_path: Path | None = None
    error: str | None = None
    ingestion_status: str | None = None
    ingestion_run_id: str | None = None
    fetched_count: int = 0
    new_post_count: int = 0
    duplicate_count: int = 0
    pagination_count: int = 0
    list_ingestion_results: tuple[dict[str, Any], ...] = ()

    @property
    def success(self) -> bool:
        return self.status in ("succeeded", "partial", "no_new_posts")

    @property
    def partial(self) -> bool:
        return self.status == "partial"

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.finished_at - self.started_at).total_seconds())

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "status": self.status,
            "run_id": self.run_id,
            "success": self.success,
            "partial": self.partial,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "list_urls": list(self.list_urls),
            "completed_lists": list(self.completed_lists),
            "list_failures": [asdict(item) for item in self.list_failures],
            "tweet_count": self.tweet_count,
            "link_count": self.link_count,
            "report_path": str(self.report_path) if self.report_path else None,
            "history_path": str(self.history_path) if self.history_path else None,
            "error": self.error,
            "ingestion_status": self.ingestion_status,
            "ingestion_run_id": self.ingestion_run_id,
            "fetched_count": self.fetched_count,
            "new_post_count": self.new_post_count,
            "duplicate_count": self.duplicate_count,
            "pagination_count": self.pagination_count,
            "list_ingestion_results": list(self.list_ingestion_results),
        }
        if include_payload:
            value.update(
                {
                    "tweets": self.tweets,
                    "aggregated": self.aggregated,
                    "summary": self.summary,
                }
            )
        return value


class TaskProviderError(RuntimeError):
    """Provider setup, login, or response error."""


class TaskSummaryError(RuntimeError):
    """Summary provider failed or returned the legacy error-string shape."""


def legacy_aggregate_by_links(tweets: list[Tweet]) -> AggregatedDigest:
    """Fallback copy of the existing application ranking algorithm.

    A legacy fetcher's own ``aggregate_by_links`` method always takes priority.
    Keeping the fallback byte-for-byte equivalent in behavior avoids changing
    ranking while allowing a minimal provider mock to participate in a run.
    """

    by_link: defaultdict[str, list[Tweet]] = defaultdict(list)
    no_links: list[Tweet] = []
    for tweet in tweets:
        if tweet.get("links"):
            for link in tweet["links"]:
                if "t.co/" in link.lower():
                    continue
                by_link[link].append(tweet)
        else:
            no_links.append(tweet)

    def score(link_tweets: list[Tweet]) -> float:
        base = sum(
            tweet["likes"]
            + (tweet["retweets"] * 1.5)
            + (tweet["replies"] * 2.0)
            + tweet["quotes"]
            + tweet["bookmarks"]
            for tweet in link_tweets
        )
        unique_authors = len({tweet["author"] for tweet in link_tweets})
        return base * unique_authors

    sorted_links = sorted(by_link.items(), key=lambda item: score(item[1]), reverse=True)
    per_author_cap = 2
    author_counts: dict[str, int] = {}
    capped: list[tuple[str, list[Tweet]]] = []
    overflow: list[tuple[str, list[Tweet]]] = []
    for item in sorted_links:
        _, link_tweets = item
        authors = {tweet["author"] for tweet in link_tweets}
        if len(authors) > 1:
            capped.append(item)
        else:
            sole_author = next(iter(authors))
            count = author_counts.get(sole_author, 0)
            if count < per_author_cap:
                author_counts[sole_author] = count + 1
                capped.append(item)
            else:
                overflow.append(item)

    return {"by_link": capped + overflow, "no_links": no_links}


_HISTORY_LOCKS_GUARD = threading.Lock()
_HISTORY_LOCKS: dict[str, threading.Lock] = {}


def _history_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path.resolve()))
    with _HISTORY_LOCKS_GUARD:
        return _HISTORY_LOCKS.setdefault(key, threading.Lock())


def _write_history_atomic(history_path: Path, filename: str, metadata: Mapping[str, Any]) -> None:
    """Update history atomically and serialize writers within this process."""

    history_path.parent.mkdir(parents=True, exist_ok=True)
    lock = _history_lock(history_path)
    with lock:
        history: dict[str, Any] = {}
        if history_path.exists():
            try:
                loaded = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    history = loaded
                else:
                    logger.warning("History index is not an object; rebuilding it")
            except (OSError, UnicodeError, json.JSONDecodeError):
                logger.warning("History index is unreadable; rebuilding it")
        history[filename] = dict(metadata)
        atomic_write_json(history_path, history)


def _remove_history_entry_atomic(history_path: Path, filename: str, run_id: str) -> None:
    """Remove only the history entry published by ``run_id``.

    Report lease fencing can discover a lost claim immediately after the
    history write.  Matching the durable run identifier prevents the old owner
    from deleting a replacement entry written by the new owner.
    """

    if not history_path.is_file():
        return
    lock = _history_lock(history_path)
    with lock:
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if not isinstance(loaded, dict):
            return
        current = loaded.get(filename)
        if not isinstance(current, dict) or str(current.get("run_id") or "") != run_id:
            return
        del loaded[filename]
        atomic_write_json(history_path, loaded)


def _publish_report_atomic(staging_path: Path, final_path: Path) -> None:
    """Publish a claim-scoped report without overwriting another owner.

    Both paths are created in the same output directory.  A hard link is an
    atomic create-if-absent operation on supported local filesystems, unlike
    ``os.replace`` which could overwrite a report already published by a newer
    claim.  Failing closed is safer than a non-atomic copy fallback.
    """

    try:
        os.link(staging_path, final_path)
    except FileExistsError as exc:
        raise TaskProviderError(
            f"Refusing to overwrite existing report: {final_path.name}"
        ) from exc
    except OSError as exc:
        raise TaskProviderError("Could not atomically publish the report") from exc
    try:
        staging_path.unlink()
    except OSError:
        logger.warning("Published report staging file could not be removed")


async def _invoke(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call sync or async injected code without blocking the event loop."""

    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)
    value = await asyncio.to_thread(partial(function, *args, **kwargs))
    if inspect.isawaitable(value):
        return await value
    return value


def _factory_value(factory: ProviderFactory, dependency: Any | None = None) -> Any:
    """Call a zero-argument factory, or a one-argument report factory."""

    if dependency is not None:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
                and parameter.default is parameter.empty
            ]
            if positional:
                return factory(dependency)
    return factory()


class DigestTaskRunner:
    """Orchestrate one digest with injectable providers and progress events."""

    def __init__(
        self,
        *,
        x_provider: XProvider | None = None,
        fetcher: Any | None = None,
        fetcher_factory: ProviderFactory | None = None,
        summary_provider: SummaryProvider | None = None,
        summary_provider_factory: ProviderFactory | None = None,
        report_generator: ReportGenerator | Callable[..., Any] | None = None,
        report_generator_factory: ProviderFactory | None = None,
        aggregator: Aggregator | None = None,
        progress_callback: ProgressCallback | None = None,
        max_concurrency: int = 4,
        now_factory: Callable[[], datetime] = datetime.now,
        run_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex[:8],
        sensitive_values: Sequence[str] = (),
        fetch_delay_factory: FetchDelayFactory | None = None,
        post_store: PostStore | None = None,
        ingestion_service: IncrementalIngestionService | None = None,
    ) -> None:
        direct_x = [value is not None for value in (x_provider, fetcher)]
        if sum(direct_x) > 1:
            raise ValueError("Pass either x_provider or fetcher, not both")
        if any(direct_x) and fetcher_factory is not None:
            raise ValueError("Pass a direct X provider/fetcher or fetcher_factory, not both")
        if summary_provider is not None and summary_provider_factory is not None:
            raise ValueError("Pass summary_provider or summary_provider_factory, not both")
        if report_generator is not None and report_generator_factory is not None:
            raise ValueError("Pass report_generator or report_generator_factory, not both")
        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
            or max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer")

        self._x_provider = x_provider or fetcher
        self._fetcher_factory = fetcher_factory
        self._summary_provider = summary_provider
        self._summary_provider_factory = summary_provider_factory
        self._report_generator = report_generator
        self._report_generator_factory = report_generator_factory
        self._aggregator = aggregator
        self._progress_callback = progress_callback
        self._max_concurrency = max_concurrency
        self._now = now_factory
        self._run_id_factory = run_id_factory
        self._sensitive_values = tuple(value for value in sensitive_values if value)
        self._fetch_delay_factory = fetch_delay_factory
        self._post_store = post_store
        self._ingestion_service = ingestion_service

    def _safe_error(self, error: Any) -> str:
        """Return a provider error with known credentials redacted."""
        return redact_sensitive_text(str(error), secrets=self._sensitive_values)

    def _resolve_x_provider(self) -> Any:
        provider = self._x_provider
        if provider is None and self._fetcher_factory is not None:
            provider = _factory_value(self._fetcher_factory)
        if provider is None:
            raise TaskProviderError("No X provider/fetcher was supplied")
        if not (
            callable(getattr(provider, "fetch_list_tweets", None))
            or callable(getattr(provider, "fetch_page", None))
        ):
            raise TaskProviderError(
                "X provider must implement fetch_list_tweets() or fetch_page()"
            )
        return provider

    def _resolve_summary_provider(self) -> Any:
        provider = self._summary_provider
        if provider is None and self._summary_provider_factory is not None:
            provider = _factory_value(self._summary_provider_factory)
        if provider is None or not callable(getattr(provider, "summarize", None)):
            raise TaskProviderError("Summary provider must implement summarize()")
        return provider

    def _resolve_report_generator(self, x_provider: Any) -> Any:
        generator = self._report_generator
        if generator is None and self._report_generator_factory is not None:
            generator = _factory_value(self._report_generator_factory, x_provider)
        if generator is None and callable(getattr(x_provider, "generate_html_report", None)):
            generator = x_provider
        if generator is None:
            raise TaskProviderError(
                "No report generator was supplied and the X provider does not implement "
                "generate_html_report()"
            )
        return generator

    async def _emit(self, event: TaskProgress) -> None:
        if self._progress_callback is None:
            return
        try:
            value = self._progress_callback(event)
            if inspect.isawaitable(value):
                await value
        except Exception:
            # Observers must not turn a completed digest into a failed digest.
            logger.warning("Progress callback failed (%s)", type(self._progress_callback).__name__)
            return

    def _set_ingestion_report_status(
        self,
        store: PostStore,
        run_id: str,
        status: str,
        error: str | None = None,
        *,
        expected_report_status: str | None = None,
        expected_report_claim_id: str | None | object = None,
    ) -> None:
        """Update report state without changing ingestion success/checkpoints."""

        run = store.get_run(run_id)
        if run is None:
            return
        previous_status = run.report_status
        previous_claim_id = run.report_claim_id
        run.report_status = status
        run.report_error = self._safe_error(error) if error else None
        if status != "generating":
            run.report_claim_id = None
            run.report_claimed_at = None
        if expected_report_status is None:
            expected_report_status = previous_status
            expected_report_claim_id = previous_claim_id
        store.update_run(
            run,
            expected_report_status=expected_report_status,
            expected_report_claim_id=expected_report_claim_id,
        )

    def _claim_ingestion_report(self, store: PostStore, run_id: str) -> IngestionRun:
        run = store.get_run(run_id)
        if run is None:
            raise TaskProviderError(f"Ingestion run {run_id!r} was not found")
        previous = run.report_status
        if previous == "succeeded":
            raise TaskProviderError("This ingestion run already has a completed report")
        if previous == "generating" and not report_claim_is_stale(
            run, now=_aware_datetime(self._now())
        ):
            raise TaskProviderError("This ingestion run report is already being generated")
        if previous not in {
            "not_started",
            "failed",
            "skipped_fetch_only",
            "generating",
        }:
            raise TaskProviderError(
                f"Ingestion run report cannot be generated from status {previous!r}"
            )
        previous_claim_id = run.report_claim_id
        run.report_status = "generating"
        run.report_error = None
        run.report_claim_id = uuid.uuid4().hex
        run.report_claimed_at = _aware_datetime(self._now())
        try:
            store.update_run(
                run,
                expected_report_status=previous,
                expected_report_claim_id=previous_claim_id,
            )
        except StorageConflictError as exc:
            raise TaskProviderError(
                "Another process claimed this ingestion report"
            ) from exc
        return run

    def _refresh_ingestion_report_claim(
        self,
        store: PostStore,
        run_id: str,
        claim_id: str,
    ) -> IngestionRun:
        """Renew and fence a report lease before publishing side effects."""

        run = store.get_run(run_id)
        if (
            run is None
            or run.report_status != "generating"
            or run.report_claim_id != claim_id
        ):
            raise StorageConflictError("Report claim ownership was lost")
        run.report_claimed_at = _aware_datetime(self._now())
        store.update_run(
            run,
            expected_report_status="generating",
            expected_report_claim_id=claim_id,
        )
        return run

    async def _run_incremental(self, request: DigestTaskRequest) -> DigestTaskResult:
        """Run the stage-1 ingestion path, then optionally the legacy digest."""

        started_at = self._now()
        requested_run_id = request.resume_ingestion_run_id
        run_id = (
            str(requested_run_id)
            if requested_run_id
            else (str(self._run_id_factory())[:16] or uuid.uuid4().hex[:8])
        )
        completed_lists: list[str] = []
        failures: list[ListFetchFailure] = []
        all_tweets: list[Tweet] = []
        aggregated: AggregatedDigest = {}
        summary = ""
        report_path: Path | None = None
        history_path: Path | None = None
        ingestion_result: IngestionResult | None = None
        ingestion_run: IngestionRun | None = None
        list_result_dicts: tuple[dict[str, Any], ...] = ()
        pagination_count = 0
        last_percent = 0
        x_provider: Any | None = None
        recoverable_interrupted_run = False
        report_claimed = False
        report_claim_id: str | None = None
        report_heartbeat_stop = asyncio.Event()
        report_heartbeat_task: asyncio.Task[None] | None = None
        report_staging_path: Path | None = None
        report_file_created = False
        history_entry_written = False
        artifacts_committed = False
        store: PostStore = self._post_store or FilePostStore(request.data_dir)

        async def report_heartbeat() -> None:
            while not report_heartbeat_stop.is_set():
                try:
                    await asyncio.wait_for(
                        report_heartbeat_stop.wait(),
                        timeout=REPORT_CLAIM_HEARTBEAT_SECONDS,
                    )
                    return
                except TimeoutError:
                    pass
                if report_claim_id is None:
                    return
                try:
                    await asyncio.to_thread(
                        self._refresh_ingestion_report_claim,
                        store,
                        run_id,
                        report_claim_id,
                    )
                except StorageConflictError:
                    logger.warning("[%s] Report heartbeat lost claim ownership", run_id)
                    return
                except Exception:
                    logger.warning("[%s] Report heartbeat could not be persisted", run_id)

        async def stop_report_heartbeat() -> None:
            nonlocal report_heartbeat_task
            report_heartbeat_stop.set()
            if report_heartbeat_task is None:
                return
            report_heartbeat_task.cancel()
            try:
                await report_heartbeat_task
            except asyncio.CancelledError:
                pass
            report_heartbeat_task = None

        async def progress(
            phase: str,
            percent: int,
            message: str,
            *,
            current_list: str | None = None,
        ) -> None:
            nonlocal last_percent
            last_percent = max(last_percent, min(100, max(0, percent)))
            await self._emit(
                TaskProgress(
                    phase=phase,
                    percent=last_percent,
                    message=message,
                    run_id=run_id,
                    completed_lists=len(completed_lists) + len(failures),
                    total_lists=len(request.list_urls),
                    failed_lists=len(failures),
                    current_list=current_list,
                    timestamp=self._now(),
                )
            )

        def build_result(status: TaskStatus, *, error: str | None = None) -> DigestTaskResult:
            run = ingestion_run
            return DigestTaskResult(
                status=status,
                started_at=started_at,
                finished_at=self._now(),
                list_urls=request.list_urls,
                run_id=run_id,
                completed_lists=tuple(completed_lists),
                list_failures=tuple(failures),
                tweet_count=len(all_tweets),
                link_count=len(aggregated.get("by_link", [])),
                tweets=all_tweets,
                aggregated=aggregated,
                summary=summary,
                report_path=report_path,
                history_path=history_path,
                error=error,
                ingestion_status=run.status if run else None,
                ingestion_run_id=run.run_id if run else run_id,
                fetched_count=run.fetched_count if run else 0,
                new_post_count=run.new_post_count if run else 0,
                duplicate_count=run.duplicate_count if run else 0,
                pagination_count=pagination_count,
                list_ingestion_results=list_result_dicts,
            )

        try:
            await progress("initializing", 0, "Resolving incremental ingestion")
            if requested_run_id:
                # The configured fetcher may also own the legacy HTML report
                # renderer.  Resolve it for rendering, but never call X while
                # replaying already persisted posts.
                if self._x_provider is not None or self._fetcher_factory is not None:
                    x_provider = self._resolve_x_provider()
                ingestion_run = store.get_run(requested_run_id)
                if ingestion_run is None:
                    raise TaskProviderError(
                        f"Ingestion run {requested_run_id!r} was not found"
                    )
                if ingestion_run.status == "running" and ingestion_run_is_stale(
                    ingestion_run, now=_aware_datetime(self._now())
                ):
                    stale_owner = ingestion_run.ingestion_owner_id
                    stale_heartbeat = ingestion_run.ingestion_heartbeat_at
                    ingestion_run.status = "failed"
                    ingestion_run.finished_at = _aware_datetime(self._now())
                    ingestion_run.ingestion_heartbeat_at = ingestion_run.finished_at
                    ingestion_run.ingestion_owner_id = None
                    ingestion_run.errors.append(
                        {
                            "kind": "stale_ingestion_recovered",
                            "message": "Ingestion owner lease expired before finalization",
                        }
                    )
                    try:
                        store.update_ingestion_run(
                            ingestion_run,
                            expected_ingestion_owner_id=stale_owner,
                            expected_ingestion_heartbeat_at=stale_heartbeat,
                        )
                    except StorageOwnershipConflictError as exc:
                        raise TaskProviderError(
                            "Ingestion run became active while recovery was attempted"
                        ) from exc
                if ingestion_run.report_status == "succeeded":
                    raise TaskProviderError(
                        "This ingestion run report is already complete"
                    )
                if ingestion_run.report_status == "generating" and not report_claim_is_stale(
                    ingestion_run, now=_aware_datetime(self._now())
                ):
                    raise TaskProviderError(
                        "This ingestion run report is already in progress"
                    )
                completed_statuses = {
                    "success",
                    "partial_success",
                    "no_new_posts",
                }
                recoverable_interrupted_run = bool(
                    ingestion_run.status == "failed"
                    and ingestion_run.new_post_ids
                    and (
                        ingestion_run.report_status in {"not_started", "failed"}
                        or report_claim_is_stale(
                            ingestion_run, now=_aware_datetime(self._now())
                        )
                    )
                )
                if (
                    ingestion_run.status not in completed_statuses
                    and not recoverable_interrupted_run
                ):
                    raise TaskProviderError("Only a completed ingestion run can be resumed")
                wanted = set(ingestion_run.new_post_ids)
                stored_posts = {post.id: post for post in store.read_posts()}
                missing = wanted.difference(stored_posts)
                if missing:
                    raise TaskProviderError(
                        "Stored ingestion posts are incomplete; refusing report regeneration"
                    )
                canonical_posts = [stored_posts[post_id] for post_id in ingestion_run.new_post_ids]
                completed_lists.extend(ingestion_run.successful_lists)
                failures.extend(
                    ListFetchFailure(
                        list_url=item,
                        error="Previous ingestion for this List failed",
                    )
                    for item in ingestion_run.failed_lists
                )
                if recoverable_interrupted_run:
                    accounted_for = set(ingestion_run.successful_lists).union(
                        ingestion_run.failed_lists
                    )
                    failures.extend(
                        ListFetchFailure(
                            list_url=item,
                            error="Ingestion was interrupted before this List completed",
                        )
                        for item in ingestion_run.requested_lists
                        if item not in accounted_for
                    )
                await progress("ingesting", 65, "Loaded persisted posts for report retry")
            else:
                x_provider = self._resolve_x_provider()
                await progress("authenticating", 5, "Checking X provider")
                page_provider = callable(getattr(x_provider, "fetch_page", None))
                requires_page_login = bool(
                    getattr(x_provider, "requires_login_for_fetch_page", False)
                )
                login = (
                    None
                    if page_provider and not requires_page_login
                    else getattr(x_provider, "login", None)
                )
                if callable(login):
                    login_result = await _invoke(login)
                    if not (
                        isinstance(login_result, tuple)
                        and len(login_result) >= 2
                        and bool(login_result[0])
                    ):
                        message = (
                            str(login_result[1])
                            if isinstance(login_result, tuple) and len(login_result) >= 2
                            else "X provider login failed"
                        )
                        raise TaskProviderError(self._safe_error(message))

                async def ingestion_progress(event: dict[str, Any]) -> None:
                    pages = int(event.get("page_number", 0) or 0)
                    percent = min(64, 10 + pages * 3)
                    await progress(
                        "ingesting",
                        percent,
                        (
                            f"List {event.get('list_id', '')}: page {pages}, "
                            f"{event.get('new_count', 0)} new"
                        ),
                        current_list=str(event.get("list_id") or "") or None,
                    )

                service = self._ingestion_service or IncrementalIngestionService(
                    provider=x_provider,
                    store=store,
                    page_size=request.page_size,
                    max_pages=request.max_pages,
                    initial_fetch_limit=request.initial_fetch_limit,
                    progress_callback=ingestion_progress,
                    sensitive_values=self._sensitive_values,
                )
                ingestion_result = await service.ingest(
                    request.list_urls,
                    run_id=run_id,
                    max_tweets=request.max_tweets,
                )
                ingestion_run = ingestion_result.run
                canonical_posts = ingestion_result.new_posts
                list_result_dicts = tuple(
                    item.to_dict() for item in ingestion_result.list_results
                )
                pagination_count = sum(
                    item.page_count for item in ingestion_result.list_results
                )
                completed_lists.extend(ingestion_run.successful_lists)
                for item in ingestion_result.list_results:
                    if not item.success:
                        failures.append(
                            ListFetchFailure(
                                list_url=item.list_id,
                                error=item.error or item.stop_reason or "Ingestion failed",
                            )
                        )

            all_tweets = [post.to_legacy_tweet() for post in canonical_posts]
            if ingestion_run is None:
                raise TaskProviderError("Ingestion did not produce a run record")
            if ingestion_run.status == "failed" and not recoverable_interrupted_run:
                detail = "; ".join(
                    f"{item.list_url}: {item.error}" for item in failures
                ) or "All requested Lists failed"
                await progress("failed", last_percent, detail)
                return build_result("failed", error=detail)

            if request.fetch_only:
                try:
                    self._set_ingestion_report_status(
                        store, run_id, "skipped_fetch_only"
                    )
                except StorageConflictError:
                    logger.info(
                        "[%s] Fetch-only report state was superseded by a report claim",
                        run_id,
                    )
                final_status: TaskStatus
                if ingestion_run.status == "partial_success":
                    final_status = "partial"
                elif ingestion_run.status == "no_new_posts":
                    final_status = "no_new_posts"
                else:
                    final_status = "succeeded"
                await progress("complete", 100, "Fetch-only ingestion complete")
                return build_result(final_status)

            if not all_tweets:
                try:
                    self._set_ingestion_report_status(
                        store, run_id, "skipped_no_new_posts"
                    )
                except StorageConflictError:
                    logger.info(
                        "[%s] No-new-post report state was superseded",
                        run_id,
                    )
                final_status = (
                    "partial" if ingestion_run.status == "partial_success" else "no_new_posts"
                )
                await progress("complete", 100, "No new posts; summary skipped")
                return build_result(final_status)

            ingestion_run = self._claim_ingestion_report(store, run_id)
            report_claimed = True
            report_claim_id = ingestion_run.report_claim_id
            if report_claim_id is None:
                raise TaskProviderError("Report claim did not receive an owner identifier")
            report_heartbeat_task = asyncio.create_task(report_heartbeat())

            await progress("aggregating", 68, f"Aggregating {len(all_tweets)} new post(s)")
            aggregate = self._aggregator
            if aggregate is None and x_provider is not None:
                aggregate = getattr(x_provider, "aggregate_by_links", None)
            aggregate = aggregate or legacy_aggregate_by_links
            aggregated_value = await _invoke(aggregate, all_tweets)
            if not isinstance(aggregated_value, dict):
                raise TypeError("Aggregator must return a dictionary")
            aggregated = aggregated_value

            summary_provider = self._resolve_summary_provider()
            await progress("summarizing", 78, "Generating summary")
            summary_value = await _invoke(summary_provider.summarize, aggregated)
            if not isinstance(summary_value, str):
                raise TaskSummaryError(
                    f"Summary provider returned {type(summary_value).__name__}, expected str"
                )
            summary = summary_value
            if summary.lstrip().lower().startswith("error"):
                raise TaskSummaryError(summary)

            report_generator = self._resolve_report_generator(x_provider)
            request.output_dir.mkdir(parents=True, exist_ok=True)
            filename = request.report_filename or (
                f"summary_{self._now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.html"
            )
            report_path = request.output_dir / filename
            if report_path.exists():
                raise TaskProviderError(f"Refusing to overwrite existing report: {filename}")
            report_staging_path = request.output_dir / f".report-{report_claim_id}.tmp.html"
            if report_staging_path.exists():
                raise TaskProviderError("Report claim staging path already exists")
            report_method = getattr(report_generator, "generate_html_report", None)
            if report_method is None and callable(report_generator):
                report_method = report_generator
            if not callable(report_method):
                raise TaskProviderError("Report generator must implement generate_html_report()")
            self._refresh_ingestion_report_claim(store, run_id, report_claim_id)
            await progress("reporting", 88, "Generating report")
            generated_path = await _invoke(
                report_method,
                aggregated,
                summary,
                report_staging_path,
                tweet_count=len(all_tweets),
                ai_model=request.ai_model,
            )
            if generated_path:
                returned_path = Path(generated_path)
                if returned_path.resolve() != report_staging_path.resolve():
                    raise TaskProviderError(
                        "Report generator returned a path different from the requested output"
                    )
            if not report_staging_path.exists():
                raise TaskProviderError(
                    f"Report generator did not create {report_staging_path.name}"
                )
            self._refresh_ingestion_report_claim(store, run_id, report_claim_id)
            _publish_report_atomic(report_staging_path, report_path)
            report_file_created = True

            list_info = getattr(x_provider, "list_info", {}) or {}
            final_status = "partial" if failures else "succeeded"
            metadata: dict[str, Any] = {
                "run_id": run_id,
                "name": list_info.get("name", "X List Summary"),
                "username": list_info.get("owner", "Unknown"),
                "tweets": len(all_tweets),
                "links": len(aggregated.get("by_link", [])),
                "profile_img": list_info.get("profile_image_url"),
                "members": list_info.get("member_count", 0),
                "status": final_status,
                "completed_lists": list(completed_lists),
                "failed_lists": [asdict(item) for item in failures],
                "generated_at": self._now().isoformat(),
                "ingestion_run_id": run_id,
            }
            metadata.update(request.history_metadata)
            history_path = request.output_dir / "history.json"
            await progress("history", 96, "Updating report history")
            _write_history_atomic(history_path, report_path.name, metadata)
            history_entry_written = True
            ingestion_run = self._refresh_ingestion_report_claim(
                store, run_id, report_claim_id
            )
            if recoverable_interrupted_run:
                ingestion_run.status = "partial_success" if failures else "success"
                ingestion_run.finished_at = self._now().astimezone()
                for failure in failures:
                    if failure.list_url not in ingestion_run.failed_lists:
                        ingestion_run.failed_lists.append(failure.list_url)
                    if not any(
                        item.get("list_id") == failure.list_url
                        for item in ingestion_run.errors
                    ):
                        ingestion_run.errors.append(
                            {
                                "list_id": failure.list_url,
                                "kind": "interrupted",
                                "message": failure.error,
                            }
                        )
                ingestion_run.report_status = "succeeded"
                ingestion_run.report_error = None
                ingestion_run.report_claim_id = None
                ingestion_run.report_claimed_at = None
                store.update_run(
                    ingestion_run,
                    expected_report_status="generating",
                    expected_report_claim_id=report_claim_id,
                )
            else:
                self._set_ingestion_report_status(
                    store,
                    run_id,
                    "succeeded",
                    expected_report_status="generating",
                    expected_report_claim_id=report_claim_id,
                )
            report_claimed = False
            artifacts_committed = True
            await progress("complete", 100, "Digest complete")
            logger.info(
                "[%s] Incremental digest completed status=%s fetched_count=%s "
                "new_count=%s duplicate_count=%s",
                run_id,
                final_status,
                ingestion_run.fetched_count,
                ingestion_run.new_post_count,
                ingestion_run.duplicate_count,
            )
            return build_result(final_status)
        except asyncio.CancelledError:
            await stop_report_heartbeat()
            if ingestion_run is not None and report_claimed:
                try:
                    self._set_ingestion_report_status(
                        store,
                        run_id,
                        "failed",
                        "Report generation was cancelled",
                        expected_report_status="generating",
                        expected_report_claim_id=report_claim_id,
                    )
                except StorageConflictError:
                    logger.warning(
                        "[%s] Cancelled report claim was superseded by another process",
                        run_id,
                    )
                except Exception:
                    logger.error("[%s] Could not persist report cancellation", run_id)
            if (
                not artifacts_committed
                and history_entry_written
                and history_path is not None
                and report_path is not None
            ):
                try:
                    await asyncio.to_thread(
                        _remove_history_entry_atomic,
                        history_path,
                        report_path.name,
                        run_id,
                    )
                except Exception:
                    logger.warning("[%s] Could not remove cancelled history entry", run_id)
            if not artifacts_committed and report_file_created and report_path is not None:
                try:
                    report_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("[%s] Could not remove cancelled report file", run_id)
            if not artifacts_committed and report_staging_path is not None:
                try:
                    report_staging_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("[%s] Could not remove cancelled report staging file", run_id)
            raise
        except Exception as exc:
            await stop_report_heartbeat()
            safe_error = self._safe_error(exc)
            if ingestion_run is not None and report_claimed:
                try:
                    self._set_ingestion_report_status(
                        store,
                        run_id,
                        "failed",
                        safe_error,
                        expected_report_status="generating",
                        expected_report_claim_id=report_claim_id,
                    )
                except StorageConflictError:
                    logger.warning(
                        "[%s] Report failure status was superseded by another process",
                        run_id,
                    )
                except Exception:
                    logger.error("[%s] Could not persist report failure status", run_id)
            if (
                not artifacts_committed
                and history_entry_written
                and history_path is not None
                and report_path is not None
            ):
                try:
                    await asyncio.to_thread(
                        _remove_history_entry_atomic,
                        history_path,
                        report_path.name,
                        run_id,
                    )
                except Exception:
                    logger.warning("[%s] Could not remove failed history entry", run_id)
            cleanup_path = report_path if report_file_created else None
            if not artifacts_committed and cleanup_path is not None:
                try:
                    cleanup_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("[%s] Could not remove failed report file", run_id)
            if not artifacts_committed and report_staging_path is not None:
                try:
                    report_staging_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("[%s] Could not remove failed report staging file", run_id)
            if not artifacts_committed:
                report_path = None
                history_path = None
            logger.error("[%s] Incremental digest failed (%s)", run_id, type(exc).__name__)
            await progress("failed", last_percent, safe_error)
            return build_result("failed", error=safe_error)
        finally:
            await stop_report_heartbeat()

    async def run(
        self,
        request: DigestTaskRequest | None = None,
        *,
        list_urls: Sequence[str] | None = None,
        max_tweets: int = 100,
        output_dir: str | Path = "output",
        ai_model: str = "",
        report_filename: str | None = None,
        history_metadata: Mapping[str, Any] | None = None,
    ) -> DigestTaskResult:
        """Run a digest and always return a structured non-cancellation result."""

        if request is not None and list_urls is not None:
            raise ValueError("Pass request or list_urls, not both")
        if request is None:
            request = DigestTaskRequest.from_values(
                list_urls or (),
                max_tweets=max_tweets,
                output_dir=output_dir,
                ai_model=ai_model,
                report_filename=report_filename,
                history_metadata=history_metadata,
            )

        if request.incremental:
            return await self._run_incremental(request)

        started_at = self._now()
        run_id = str(self._run_id_factory())[:16] or uuid.uuid4().hex[:8]
        total_lists = len(request.list_urls)
        completed_lists: list[str] = []
        failures: list[ListFetchFailure] = []
        all_tweets: list[Tweet] = []
        aggregated: AggregatedDigest = {}
        summary = ""
        report_path: Path | None = None
        history_path: Path | None = None
        last_percent = 0

        async def progress(
            phase: str,
            percent: int,
            message: str,
            *,
            current_list: str | None = None,
        ) -> None:
            nonlocal last_percent
            last_percent = max(last_percent, min(100, max(0, percent)))
            await self._emit(
                TaskProgress(
                    phase=phase,
                    percent=last_percent,
                    message=message,
                    run_id=run_id,
                    completed_lists=len(completed_lists) + len(failures),
                    total_lists=total_lists,
                    failed_lists=len(failures),
                    current_list=current_list,
                    timestamp=self._now(),
                )
            )

        def result(status: TaskStatus, *, error: str | None = None) -> DigestTaskResult:
            return DigestTaskResult(
                status=status,
                started_at=started_at,
                finished_at=self._now(),
                list_urls=request.list_urls,
                run_id=run_id,
                completed_lists=tuple(completed_lists),
                list_failures=tuple(failures),
                tweet_count=len(all_tweets),
                link_count=len(aggregated.get("by_link", [])),
                tweets=all_tweets,
                aggregated=aggregated,
                summary=summary,
                report_path=report_path,
                history_path=history_path,
                error=error,
            )

        try:
            logger.info("[%s] Digest task started for %s list(s)", run_id, total_lists)
            await progress("initializing", 0, "Resolving providers")
            x_provider = self._resolve_x_provider()
            summary_provider = self._resolve_summary_provider()
            report_generator = self._resolve_report_generator(x_provider)

            await progress("authenticating", 5, "Checking X provider")
            login = getattr(x_provider, "login", None)
            if callable(login):
                login_result = await _invoke(login)
                if not (
                    isinstance(login_result, tuple)
                    and len(login_result) >= 2
                    and bool(login_result[0])
                ):
                    message = (
                        str(login_result[1])
                        if isinstance(login_result, tuple) and len(login_result) >= 2
                        else "X provider login failed"
                    )
                    raise TaskProviderError(self._safe_error(message))
            await progress("fetching", 10, f"Fetching {total_lists} list(s)")

            semaphore = asyncio.Semaphore(self._max_concurrency)
            completion_lock = asyncio.Lock()
            completed_fetches = 0

            async def fetch_one(
                index: int,
                list_url: str,
            ) -> tuple[str, list[Tweet] | None, Exception | None]:
                nonlocal completed_fetches
                tweets: list[Tweet] | None = None
                failure: Exception | None = None
                try:
                    delay = 0.0
                    if self._fetch_delay_factory is not None:
                        delay = max(0.0, float(self._fetch_delay_factory(index)))
                    async with semaphore:
                        fetched = await _invoke(
                            x_provider.fetch_list_tweets,
                            list_url,
                            request.max_tweets,
                            delay=delay,
                        )
                    if fetched is None:
                        fetched = []
                    if not isinstance(fetched, list):
                        raise TypeError(f"List provider returned {type(fetched).__name__}, expected list")
                    tweets = fetched
                except Exception as exc:
                    failure = exc

                async with completion_lock:
                    completed_fetches += 1
                    if failure is None:
                        completed_lists.append(list_url)
                    else:
                        failures.append(
                            ListFetchFailure(
                                list_url=list_url,
                                error=self._safe_error(failure),
                            )
                        )
                    fetch_percent = 10 + int(50 * completed_fetches / total_lists)
                await progress(
                    "fetching",
                    fetch_percent,
                    f"Fetched {completed_fetches}/{total_lists} list(s)",
                    current_list=list_url,
                )
                return list_url, tweets, failure

            outcomes = await asyncio.gather(
                *(fetch_one(index, url) for index, url in enumerate(request.list_urls))
            )
            for _, tweets, failure in outcomes:
                if failure is None and tweets is not None:
                    all_tweets.extend(tweets)

            if not all_tweets:
                detail = "; ".join(f"{item.list_url}: {item.error}" for item in failures)
                raise TaskProviderError(detail or "No tweets were fetched from the requested lists")

            await progress("aggregating", 65, f"Aggregating {len(all_tweets)} tweet(s)")
            aggregate = self._aggregator or getattr(x_provider, "aggregate_by_links", None)
            if aggregate is None:
                aggregate = legacy_aggregate_by_links
            aggregated_value = await _invoke(aggregate, all_tweets)
            if not isinstance(aggregated_value, dict):
                raise TypeError("Aggregator must return a dictionary")
            aggregated = aggregated_value
            await progress(
                "aggregating",
                72,
                f"Found {len(aggregated.get('by_link', []))} shared link(s)",
            )

            await progress("summarizing", 78, "Generating summary")
            summary_value = await _invoke(summary_provider.summarize, aggregated)
            if not isinstance(summary_value, str):
                raise TaskSummaryError(
                    f"Summary provider returned {type(summary_value).__name__}, expected str"
                )
            summary = summary_value
            if summary.lstrip().lower().startswith("error"):
                raise TaskSummaryError(summary)
            await progress("summarizing", 85, "Summary complete")

            request.output_dir.mkdir(parents=True, exist_ok=True)
            filename = request.report_filename or (
                f"summary_{self._now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.html"
            )
            candidate_report_path = request.output_dir / filename
            if candidate_report_path.exists():
                raise TaskProviderError(f"Refusing to overwrite existing report: {filename}")
            report_path = candidate_report_path
            report_method = getattr(report_generator, "generate_html_report", None)
            if report_method is None and callable(report_generator):
                report_method = report_generator
            if not callable(report_method):
                raise TaskProviderError("Report generator must implement generate_html_report()")

            await progress("reporting", 90, "Generating report")
            generated_path = await _invoke(
                report_method,
                aggregated,
                summary,
                report_path,
                tweet_count=len(all_tweets),
                ai_model=request.ai_model,
            )
            if generated_path:
                report_path = Path(generated_path)
            if not report_path.exists():
                raise TaskProviderError(f"Report generator did not create {report_path}")
            await progress("reporting", 94, f"Report written to {report_path.name}")

            list_info = getattr(x_provider, "list_info", {}) or {}
            status: TaskStatus = "partial" if failures else "succeeded"
            metadata: dict[str, Any] = {
                "run_id": run_id,
                "name": list_info.get("name", "X List Summary"),
                "username": list_info.get("owner", "Unknown"),
                "tweets": len(all_tweets),
                "links": len(aggregated.get("by_link", [])),
                "profile_img": list_info.get("profile_image_url"),
                "members": list_info.get("member_count", 0),
                "status": status,
                "completed_lists": list(completed_lists),
                "failed_lists": [asdict(item) for item in failures],
                "generated_at": self._now().isoformat(),
            }
            metadata.update(request.history_metadata)
            history_path = request.output_dir / "history.json"
            await progress("history", 97, "Updating report history")
            await asyncio.to_thread(_write_history_atomic, history_path, report_path.name, metadata)
            await progress("complete", 100, "Digest complete")
            logger.info(
                "[%s] Digest task completed: status=%s tweets=%s links=%s",
                run_id,
                status,
                len(all_tweets),
                len(aggregated.get("by_link", [])),
            )
            return result(status)
        except Exception as exc:
            safe_error = self._safe_error(exc)
            logger.error("[%s] Digest task failed (%s)", run_id, type(exc).__name__)
            await progress("failed", last_percent, safe_error)
            return result("failed", error=safe_error)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an injectable X list digest task")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run entirely offline with deterministic mock providers",
    )
    parser.add_argument(
        "--list",
        dest="lists",
        action="append",
        help="mock list identifier; repeat for more than one list",
    )
    parser.add_argument("--max-tweets", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="persist incremental posts and checkpoints without calling an LLM",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="use durable incremental ingestion before generating the mock digest",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--initial-fetch-limit", type=int, default=100)
    parser.add_argument(
        "--resume-ingestion-run",
        help="regenerate a report from posts already stored for this ingestion run",
    )
    parser.add_argument(
        "--fail-list",
        action="append",
        default=[],
        help="make a named mock list fail (for partial-failure demos)",
    )
    return parser


async def _mock_main(args: argparse.Namespace) -> DigestTaskResult:
    events: list[TaskProgress] = []

    def show_progress(event: TaskProgress) -> None:
        events.append(event)
        print(f"[{event.percent:3d}%] {event.phase}: {event.message}")

    use_incremental = bool(
        args.incremental or args.fetch_only or args.resume_ingestion_run
    )
    if use_incremental:
        failures = {
            (list_id, 1): RuntimeError(f"Mock list failure: {list_id}")
            for list_id in args.fail_list
        }
        x_provider: Any = MockXIngestionProvider(page_failures=failures)
        store: PostStore | None = FilePostStore(args.data_dir)
    else:
        x_provider = MockXProvider(fail_lists=args.fail_list)
        store = None
    runner = DigestTaskRunner(
        x_provider=x_provider,
        summary_provider=MockSummaryProvider(),
        report_generator=MockReportGenerator(),
        progress_callback=show_progress,
        post_store=store,
    )
    request = DigestTaskRequest.from_values(
        args.lists or ["mock://mixed"],
        max_tweets=args.max_tweets,
        output_dir=args.output_dir,
        ai_model="MockSummaryProvider",
        incremental=use_incremental,
        fetch_only=args.fetch_only,
        data_dir=args.data_dir,
        page_size=args.page_size,
        max_pages=args.max_pages,
        initial_fetch_limit=args.initial_fetch_limit,
        resume_ingestion_run_id=args.resume_ingestion_run,
    )
    return await runner.run(request)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.mock:
        parser.error(
            "Only the dependency-free --mock entry is configured here. "
            "Inject the existing XListFetcher/XApiFetcher and LLMProvider from application code."
        )
    result = asyncio.run(_mock_main(args))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DigestTaskRequest",
    "DigestTaskResult",
    "DigestTaskRunner",
    "ListFetchFailure",
    "REPORT_CLAIM_LEASE",
    "TaskProgress",
    "legacy_aggregate_by_links",
    "main",
    "report_claim_is_stale",
]
