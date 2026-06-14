from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = Field(..., alias="DATABASE_URL")
    # SQLAlchemy pool sizing (PER worker process). Kept small to bound the
    # shared Supabase session-mode pool across the worker fleet — db.py floors
    # pool_size at worker_concurrency + 1 so a held job session can never starve
    # a sibling slot regardless of these values. max_overflow absorbs the
    # transient claim/heartbeat sessions on top. pool_pre_ping (db.py) discards
    # connections the pooler silently dropped under a long-held job session.
    db_pool_size: int = Field(2, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(4, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(10, alias="DB_POOL_TIMEOUT")
    db_pool_recycle: int = Field(600, alias="DB_POOL_RECYCLE")
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_service_role_key: str = Field(..., alias="SUPABASE_SERVICE_ROLE_KEY")
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")

    deployments_bucket: str = Field("deployments", alias="DEPLOYMENTS_BUCKET")
    drive_bucket: str = Field("drive", alias="DRIVE_BUCKET")
    # Public bucket where marketplace/cross-workspace job outputs are dropped on
    # completion. Must be configured as PUBLIC in Supabase Storage.
    job_outputs_bucket: str = Field("job-outputs", alias="JOB_OUTPUTS_BUCKET")
    # Hard cap on total bytes uploaded from a cross-workspace job's tmp drive.
    # Protects against abuse / runaway disk usage in the public bucket.
    job_outputs_max_bytes: int = Field(
        500 * 1024 * 1024, alias="JOB_OUTPUTS_MAX_BYTES"
    )

    # Deployment cache root — extracted bundles and venvs live under here
    deployments_root: str = Field(
        "/var/puras/deployments", alias="DEPLOYMENTS_ROOT"
    )
    # Dev mode: if set, the worker uses this local dir as a "virtual deployment"
    # for any job whose deployment_id is null. Lets you edit code and re-run
    # without push/upload cycles.
    local_project_path: str | None = Field(None, alias="LOCAL_PROJECT_PATH")

    # Offline runner (`puras run --local`): the open-core local mode. When set,
    # the runtime runs with NO platform — there is no Postgres, no bucket, and no
    # platform API. The drive is a plain local directory (no bucket push/pull),
    # the agent loop runs on a LocalRunContext (platform_enabled=False), and the
    # hosted-only tools (memory / media / web / cross-skillpack subagents) are
    # switched off. Hosted leaves this False; `local_run.run_local` sets it.
    local_mode: bool = Field(False, alias="PURAS_LOCAL_MODE")

    # Drive root. A plain local directory backed by the Supabase bucket via
    # explicit upload/download (no FUSE — see worker/drive.py). LOCAL_DRIVE_PATH
    # is the host dir in dev; prod leaves it unset and falls back to
    # DRIVE_MOUNT_PATH (just a writable container dir now, no longer a mount).
    local_drive_path: str | None = Field(None, alias="LOCAL_DRIVE_PATH")
    drive_mount_path: str = Field("/drive-mount", alias="DRIVE_MOUNT_PATH")

    # Retained for compatibility / any non-drive S3 use; no longer used to mount
    # the drive (that was s3fs-fuse, now removed).
    supabase_s3_endpoint: str | None = Field(None, alias="SUPABASE_S3_ENDPOINT")
    supabase_s3_key_id: str | None = Field(None, alias="SUPABASE_S3_KEY_ID")
    supabase_s3_secret: str | None = Field(None, alias="SUPABASE_S3_SECRET")
    supabase_s3_region: str = Field("eu-west-1", alias="SUPABASE_S3_REGION")

    # Job workdir root — per-job dirs live under here.
    workdir_root: str = Field("/tmp/puras-jobs", alias="WORKDIR_ROOT")

    # Bash tool tuning
    bash_default_timeout: int = Field(60, alias="BASH_DEFAULT_TIMEOUT")
    bash_max_timeout: int = Field(600, alias="BASH_MAX_TIMEOUT")

    # Internal worker → API auth + base URL for media generation calls
    api_base: str = Field("http://localhost:8000", alias="PURAS_API_BASE")
    service_token: str = Field(..., alias="PURAS_SERVICE_TOKEN")
    # Optional path to a file holding the service token (P2-9 token refresh on
    # long runs): when set, the worker re-reads it (briefly cached) before each
    # platform callback, so a rotated mounted secret is picked up mid-run without
    # a restart. The API accepts a grace set during rotation — see
    # require_service_token. Unset = use the static PURAS_SERVICE_TOKEN.
    service_token_file: str | None = Field(None, alias="PURAS_SERVICE_TOKEN_FILE")

    # --- Confirmation gates / human-in-the-loop (P1-5) ---
    # A `confirm: true` tool pauses the run for a human approve/deny decision.
    # The worker polls for the decision this often, up to this timeout — after
    # which the gated call is auto-DENIED (a missed approval must fail closed, not
    # silently run a side effect). The wait holds the job's slot; keep the timeout
    # comfortably under MAX_AGENT_SECONDS so the run itself doesn't time out first.
    approval_timeout_s: int = Field(1800, alias="APPROVAL_TIMEOUT_S")
    approval_poll_interval_s: float = Field(3.0, alias="APPROVAL_POLL_INTERVAL_S")

    # Default agentic model when a skill omits `model:`. Public slug form
    # (see worker/worker/llm_models.py). Kept under the legacy CLAUDE_MODEL
    # env name so existing deployments don't need a secret rotation.
    default_model_slug: str = Field(
        "claude/sonnet-4-6", alias="CLAUDE_MODEL"
    )

    # --- Memory v2 embeddings (OPTIONAL) ---
    # When an API key is configured, workspace-memory writes get an embedding
    # and memory_search gains a semantic branch (pgvector). Unset = memory v2
    # runs in exact+FTS mode — fully functional, just lexical. Any
    # OpenAI-compatible /v1/embeddings endpoint works (base_url override).
    # Falls back to OPENAI_API_KEY when PURAS_EMBEDDINGS_API_KEY is unset.
    embeddings_api_key: str | None = Field(None, alias="PURAS_EMBEDDINGS_API_KEY")
    embeddings_base_url: str | None = Field(None, alias="PURAS_EMBEDDINGS_BASE_URL")
    embeddings_model: str = Field(
        "text-embedding-3-small", alias="PURAS_EMBEDDINGS_MODEL"
    )
    # Must match the `vector(N)` column from migration 032.
    embeddings_dims: int = Field(1536, alias="PURAS_EMBEDDINGS_DIMS")
    max_agent_steps: int = Field(250, alias="MAX_AGENT_STEPS")
    max_agent_seconds: int = Field(3600, alias="MAX_AGENT_SECONDS")
    # Global per-run, per-tool call cap (P2-9): a blanket safety net under any
    # skill-specific `tool_limits`. 0 = disabled (rely on per-skill caps only).
    max_tool_calls_per_run: int = Field(0, alias="MAX_TOOL_CALLS_PER_RUN")
    # Wide default so pipeline skills (deterministic Python that calls
    # `puras.subagent.run` and waits on child renders) don't trip on the
    # subprocess timeout. Tight single-purpose functions finish in
    # seconds regardless; this is just the ceiling.
    function_timeout_seconds: int = Field(1800, alias="FUNCTION_TIMEOUT_SECONDS")

    worker_id: str = Field("local", alias="FLY_MACHINE_ID")
    poll_interval_seconds: float = Field(2.0, alias="POLL_INTERVAL_SECONDS")
    # Fly region this machine runs in (for the admin fleet view). Fly sets it.
    fly_region: str | None = Field(None, alias="FLY_REGION")

    # How many jobs ONE worker process runs concurrently. Jobs are I/O-bound
    # (almost all wall time is awaiting the LLM + fal via asyncio.to_thread), so
    # N slots in one event loop genuinely overlap their waits — a single machine
    # processes up to N jobs at once, far cheaper than N machines. Each claim is
    # its own skip-locked transaction so slots never double-claim.
    # 1 = the original strict-serial behavior. This is pure throughput now:
    # same-deployment subagents run IN-PROCESS (no queued child job — see
    # agent_runner._dispatch_subagent), so a pipeline runs entirely within its
    # parent's one slot and can't deadlock regardless of N. (Only rare
    # cross-skillpack subagents still use a queued child.) Watch memory: every
    # concurrent job + any subprocess it spawns shares the machine's RAM.
    worker_concurrency: int = Field(1, alias="WORKER_CONCURRENCY")

    sentry_dsn: str | None = Field(None, alias="SENTRY_DSN")
    environment: str = Field("development", alias="ENVIRONMENT")

    # Job routing lane — must match how the API stamps jobs (see api config).
    # The worker only claims jobs on its own lane, so a local dev worker polling
    # the shared prod DB never steals real prod jobs. Defaults to "prod" in
    # production and "local" elsewhere; override with JOB_LANE for multi-dev.
    job_lane_override: str | None = Field(None, alias="JOB_LANE")

    # --- Health endpoint (HTTP /health for fly / k8s liveness checks) ---
    health_port: int = Field(8080, alias="HEALTH_PORT")
    # Worker reports unhealthy if no heartbeat in this many seconds. Default =
    # 30s. The heartbeat is refreshed by a background task every
    # heartbeat_interval_s (NOT once per job), so a single long job no longer
    # lets it go stale; only a genuinely wedged process trips the check.
    health_staleness_window_s: float = Field(30.0, alias="HEALTH_STALENESS_WINDOW_S")
    # How often the background heartbeat task refreshes liveness, decoupled from
    # job progress. Must be comfortably below health_staleness_window_s. The same
    # tick stamps jobs.heartbeat_at for the in-flight job so the API reaper can
    # detect a dead worker's stranded job.
    heartbeat_interval_s: float = Field(10.0, alias="HEARTBEAT_INTERVAL_S")
    # Shutdown drain: how long after SIGTERM in-flight jobs may keep running
    # before _drain_watchdog cancels them and hands them back to the queue for a
    # replacement machine. Must sit comfortably below fly.worker.toml's
    # kill_timeout (300s — the Fly maximum) so the requeue COMMITS before the
    # platform SIGKILLs the VM; the slack also covers task unwinding. Jobs that
    # finish within this window drain normally and are never requeued.
    drain_requeue_after_s: float = Field(240.0, alias="DRAIN_REQUEUE_AFTER_S")

    # Per-job spend ceiling in micros (1_000_000 = $1). When > 0, an agentic run
    # is aborted as soon as its accrued cost (LLM + media) reaches this cap,
    # bounding the blast radius of a single expensive run (e.g. a long video)
    # against a user's free signup credit. 0 disables the cap.
    max_job_cost_micros: int = Field(0, alias="MAX_JOB_COST_MICROS")

    # --- Exact-match prompt/response cache (P0-2a) ---
    # When on, an agent LLM call whose normalized request (model + system +
    # messages + tools + max_tokens + workspace) hashes to a stored response is
    # served from the cache with NO upstream call and billed 0 — the step-0
    # call (full system + tools + first user turn) is the common, big win on a
    # re-run of the same skill+inputs. Per-workspace; the key auto-misses on any
    # skill/model/input change, so the TTL is only a staleness backstop.
    prompt_cache_enabled: bool = Field(True, alias="PROMPT_CACHE_ENABLED")
    # Max age of a cache entry that may be served (seconds). 0 = no expiry.
    prompt_cache_ttl_s: int = Field(7 * 24 * 3600, alias="PROMPT_CACHE_TTL_S")

    # --- Memory guards (OOM resilience) ---
    # WORKER_CONCURRENCY jobs + the subprocesses they spawn (ffmpeg encodes in
    # particular) share one machine's RAM. Two guards keep a single runaway
    # encode from taking the whole machine — and every job on it — down:
    #
    # 1) Admission: a slot won't claim a NEW job while this machine is already
    #    busy AND free memory is below the floor (used% >= ceiling). An idle
    #    machine always accepts work; the gate only stops STACKING onto pressure.
    claim_mem_ceiling_pct: float = Field(90.0, alias="CLAIM_MEM_CEILING_PCT")
    # 2) Blast radius: heavy child subprocesses (the bash tool + deterministic
    #    skill functions, and any ffmpeg they fork) are marked the kernel's
    #    preferred OOM victim, so an over-budget encode is killed instead of the
    #    worker process (whose death would strand all its concurrent jobs). When
    #    > 0, those children also get a hard RLIMIT_AS cap (MB) — off by default
    #    because address-space limits are unreliable for ffmpeg/Python (large
    #    virtual reservations); the OOM-victim hint is the primary mechanism.
    child_mem_limit_mb: int = Field(0, alias="CHILD_MEM_LIMIT_MB")

    # --- Token economy: tool-result offloading (P1) ---
    # A multi-step agent re-reads its whole conversation on every turn (the
    # prompt cache makes re-reads cheap, but they're still the dominant cost),
    # so a big tool result — a fetched web page, long bash stdout, a verbose
    # subagent return — is paid for on every subsequent step it stays inline.
    # When a tool result's text exceeds this many chars, we OFFLOAD it: write
    # the full payload to a drive file and leave only a head + a `file_read`
    # pointer in context (restorable, à la Manus' "drop the page, keep the URL").
    # This is append-only, so unlike retroactive clearing it never invalidates
    # the prompt cache. 0 disables. Non-string (image/multimodal) results are
    # never offloaded.
    tool_result_offload_chars: int = Field(12000, alias="TOOL_RESULT_OFFLOAD_CHARS")
    # How much of the head to keep inline so the IMMEDIATE next turn usually has
    # what it needs without a re-read (the model reasons over a fresh result the
    # turn after it lands; only older steps pay the full re-read).
    tool_result_offload_head_chars: int = Field(6000, alias="TOOL_RESULT_OFFLOAD_HEAD_CHARS")

    # --- Token economy: server-side context editing (P2/P3) ---
    # Anthropic's `clear_tool_uses_20250919`: once the prompt exceeds `trigger`
    # input tokens, the API clears the OLDEST tool results (keeping the last
    # `keep` tool_use/result pairs), replacing each with a placeholder so the
    # model knows content was removed. Clearing invalidates the cached prefix
    # from the clear point, so `clear_at_least` makes each clear remove enough
    # tokens to be worth the one cache re-write. `clear_tool_inputs` also drops
    # the (often large) tool-call parameters of cleared turns (P3). Only the
    # Anthropic provider honors this; OpenRouter ignores it.
    context_editing_enabled: bool = Field(True, alias="CONTEXT_EDITING_ENABLED")
    context_editing_trigger_tokens: int = Field(60000, alias="CONTEXT_EDITING_TRIGGER_TOKENS")
    context_editing_keep_tool_uses: int = Field(6, alias="CONTEXT_EDITING_KEEP_TOOL_USES")
    context_editing_clear_at_least_tokens: int = Field(
        20000, alias="CONTEXT_EDITING_CLEAR_AT_LEAST_TOKENS"
    )
    context_editing_clear_tool_inputs: bool = Field(
        True, alias="CONTEXT_EDITING_CLEAR_TOOL_INPUTS"
    )

    # --- Product analytics (PostHog) ---
    # Project ingestion token (phc_...); no-op when unset. Not the phx_ key.
    posthog_api_key: str | None = Field(None, alias="POSTHOG_API_KEY")
    posthog_host: str = Field("https://us.i.posthog.com", alias="POSTHOG_HOST")

    @property
    def job_lane(self) -> str:
        """Lane this worker claims jobs on. Explicit JOB_LANE wins; otherwise
        "prod" in production, "local" everywhere else. Must match the API."""
        if self.job_lane_override:
            return self.job_lane_override
        return "prod" if self.environment == "production" else "local"


@lru_cache
def get_settings() -> WorkerSettings:
    return WorkerSettings()


# --- Service-token refresh (P2-9: token refresh on long runs) -----------------
# In-flight long runs hold the worker's platform service token in memory; if an
# operator rotates PURAS_SERVICE_TOKEN the API would 401 every callback (media /
# web / subagent) of a run that started before the roll. When PURAS_SERVICE_TOKEN
# _FILE points at a rotating mounted secret, the worker re-reads it (cached for a
# few seconds) before each callback so the new token is picked up mid-run, while
# the API accepts both old+new during the grace window (require_service_token).
_svc_tok_cache: tuple[float, str] | None = None
_SVC_TOK_TTL_S = 30.0


def service_token() -> str:
    """The worker's CURRENT platform service token. Reads PURAS_SERVICE_TOKEN_FILE
    fresh (briefly cached) when set so a rotated secret is honored without a
    restart; otherwise the static PURAS_SERVICE_TOKEN."""
    import time as _time

    global _svc_tok_cache
    s = get_settings()
    path = s.service_token_file
    if not path:
        return s.service_token
    now = _time.monotonic()
    if _svc_tok_cache is not None and now - _svc_tok_cache[0] < _SVC_TOK_TTL_S:
        return _svc_tok_cache[1]
    try:
        with open(path) as f:
            tok = f.read().strip()
    except OSError:
        tok = ""
    tok = tok or s.service_token  # fall back if the file is missing/empty
    _svc_tok_cache = (now, tok)
    return tok
