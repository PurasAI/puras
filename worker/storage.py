import mimetypes
import shutil
from pathlib import Path

import structlog

from .config import get_settings

log = structlog.get_logger()

# Drive objects we've already pushed to Supabase in local-drive dev mode,
# keyed by bucket path -> (size, mtime_ns). Lets repeated signs of an unchanged
# file skip the re-upload. Never populated in prod: local_drive_path is unset
# there, so _local_drive_file() short-circuits and this stays empty.
_local_upload_memo: dict[str, tuple[int, int]] = {}


def _client():
    # Imported lazily so the offline runner (which never reaches the bucket —
    # local_mode short-circuits upload/pull/sign) can import this module without
    # storage3 installed.
    from storage3 import create_client

    s = get_settings()
    url = f"{s.supabase_url}/storage/v1"
    return create_client(
        url,
        {
            "apiKey": s.supabase_service_role_key,
            "Authorization": f"Bearer {s.supabase_service_role_key}",
        },
        is_async=False,
    )


def download(bucket: str, path: str) -> bytes:
    return _client().from_(bucket).download(path)


def upload(bucket: str, path: str, data: bytes, content_type: str) -> None:
    _client().from_(bucket).upload(
        path=path,
        file=data,
        # cache-control is STORED object metadata: Supabase then serves the
        # blob with `cache-control: max-age=3600` instead of its `no-cache`
        # default, so browsers actually cache job media (run outputs are
        # write-once under <skill>/<jobshort>/). Pairs with the cacheable 302
        # on /v1/jobs/{id}/assets — without this every page view re-downloads.
        file_options={
            "content-type": content_type,
            "upsert": "true",
            "cache-control": "3600",
        },
    )


def public_url(bucket: str, path: str) -> str:
    """Direct URL for an object in a public bucket. No signing, no expiry."""
    s = get_settings()
    return f"{s.supabase_url}/storage/v1/object/public/{bucket}/{path}"


def _local_drive_file(bucket: str, path: str) -> Path | None:
    """Host-filesystem location of a drive-bucket object, or None for a
    non-drive bucket (or before the drive is set up).

    The worker drive is always a local dir now (see drive.py), so the object at
    bucket key `<workspace_id>/<rel>` lives at `<drive_root>/<workspace_id>/<rel>`.
    """
    s = get_settings()
    if bucket != s.drive_bucket:
        return None
    try:
        from .drive import get_drive_root

        root = get_drive_root().resolve()
    except Exception:
        # Drive not set up in this process (e.g. a non-worker import) — treat as
        # "no local file"; signing then proceeds straight against the bucket.
        return None
    full = (root / path.lstrip("/")).resolve()
    try:
        full.relative_to(root)  # never read outside the drive root
    except ValueError:
        return None
    return full


def _ensure_local_object_uploaded(bucket: str, path: str) -> bool:
    """Lazily push a locally-created drive file to the bucket before signing.

    The worker's drive writes (bash/python copy, resize, download, upscale, a
    generated output) only hit local disk, so a signed bucket URL would 404 when
    Fal fetches it. Upload the file the moment a URL is requested. Idempotent per
    (size, mtime_ns) so unchanged files aren't re-uploaded; a re-resized file
    (new mtime) is re-pushed. Returns True if it uploaded, False if there's no
    local file to push (already-remote object, or a path never created locally).
    """
    local = _local_drive_file(bucket, path)
    if local is None:
        return False
    try:
        st = local.stat()
    except OSError:
        # Not a locally-materialized file (already-remote object, or the agent
        # referenced a path it never created). Let signing proceed and surface
        # any genuine "not found" from Supabase.
        return False
    key = (st.st_size, st.st_mtime_ns)
    if _local_upload_memo.get(path) == key:
        return False
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    upload(bucket, path, local.read_bytes(), ctype)
    _local_upload_memo[path] = key
    log.info("drive_local_lazy_upload", path=path, size=st.st_size)
    return True


def _candidate_drive_paths(obj) -> list[str]:
    """Collect drive-relative paths referenced in a job's inputs.

    A file input arrives as `{drive_path: ...}` / `{path: ...}` or a bare
    relative string (URLs and data: URIs are skipped). We only treat a bare
    string as a candidate when it has a folder separator — drive uploads are
    always `uploads/<uuid>.<ext>`, so this skips scalar inputs like a language
    code or a brand name without a false download attempt."""
    out: list[str] = []

    def add(p):
        if isinstance(p, str):
            s = p.strip().lstrip("/")
            if s and ".." not in s.split("/"):
                out.append(s)

    def walk(v):
        if isinstance(v, str):
            s = v.strip()
            if s and "://" not in s and not s.startswith("data:") and "/" in s:
                add(s)
        elif isinstance(v, dict):
            dp = v.get("drive_path") or v.get("path")
            if isinstance(dp, str):
                add(dp)
            for val in v.values():
                if isinstance(val, (dict, list)):
                    walk(val)
        elif isinstance(v, list):
            for item in v:
                walk(item)

    walk(obj)
    return out


def ensure_input_files(workspace_id: str, inputs: dict) -> None:
    """Materialize a job's declared input files onto the worker's local drive
    before the skill runs, so the skill's first read is a local hit.

    Inputs are uploaded browser→Supabase directly (the API mints a signed upload
    URL; bytes never touch the worker box), so at job start they exist in the
    bucket but not on the worker's local disk. For each declared input not
    already present, pull it from the bucket and write it locally.

    Best-effort: a path that isn't a real object (a plain string input that
    merely looks path-ish) just 404s and is skipped; the skill's own read then
    raises a genuine not-found. It re-pulls each small declared input once per
    job — cheap, and the price of a consistent first read."""
    from .drive import workspace_drive

    s = get_settings()
    root = workspace_drive(workspace_id).resolve()
    for rel in _candidate_drive_paths(inputs):
        dest = (root / rel).resolve()
        try:
            dest.relative_to(root)  # never write outside the workspace drive
        except ValueError:
            continue
        if dest.exists():
            continue
        try:
            data = download(s.drive_bucket, f"{workspace_id}/{rel}")
        except Exception:
            continue  # not an object — a non-file input that looked path-ish
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        log.info("drive_input_materialized", path=f"{workspace_id}/{rel}", size=len(data))


def upload_drive_file(workspace_id: str, rel: str) -> bool:
    """Push a worker-local drive file to the bucket so the API (which serves
    ONLY from the bucket) and upstreams can read it.

    The worker now writes plain local files; nothing reaches the bucket
    implicitly the way a FUSE write did. So whenever the worker produces a file
    that must be servable — generated media shown in the pipeline, a declared
    output, a file about to be signed for Fal — it calls this to push the bytes.
    Idempotent per (size, mtime_ns): an unchanged file isn't re-pushed, a
    rewritten one (new mtime) is. Returns True if it uploaded, False if the file
    isn't present locally (nothing to push). Best-effort at call sites — a failed
    push is retried by the end-of-job sync-out.
    """
    s = get_settings()
    # Offline runner: there is no bucket. The file already lives on local disk
    # (that's the durable store locally), so there's nothing to push.
    if s.local_mode:
        return False
    rel = rel.strip().lstrip("/")
    if rel.startswith("drive/"):
        rel = rel[len("drive/") :]
    return _ensure_local_object_uploaded(s.drive_bucket, f"{workspace_id}/{rel}")


def _output_drive_paths(obj) -> list[str]:
    """Collect drive-path-looking strings anywhere in a job result.

    Unlike `_candidate_drive_paths` (tuned for inputs, where a path arrives as
    `{drive_path}`/`{path}` or a top-level string), output media are bare
    drive-path strings under semantic keys — `{"video": "out/ad.mp4"}`,
    `{"end_cards": [{"output_url": "_jobs/x.webp"}]}`. So recurse EVERY string
    value. A string with no '/', a URL scheme, or a data: URI is skipped; any
    false positive is a no-op upload (the file simply isn't local)."""
    out: list[str] = []

    def walk(v):
        if isinstance(v, str):
            s = v.strip().lstrip("/")
            if s and "://" not in s and not s.startswith("data:") and "/" in s and ".." not in s.split("/"):
                out.append(s)
        elif isinstance(v, dict):
            for val in v.values():
                walk(val)
        elif isinstance(v, list):
            for item in v:
                walk(item)

    walk(obj)
    return out


def sync_output_files(workspace_id: str, result) -> None:
    """End-of-job: push every worker-local drive file referenced in the result
    to the bucket, so the API (which serves only from the bucket) can return the
    job's outputs.

    `generate_*` / `download_url` already pushed theirs eagerly; this is the
    backstop for outputs a skill produced another way — a `stitch`/ffmpeg render,
    a `.py` tool, raw bash. Walks the result for drive paths and uploads each
    (idempotent, so eagerly-pushed files are no-ops). Best-effort per file.
    """
    for rel in _output_drive_paths(result):
        try:
            upload_drive_file(workspace_id, rel)
        except Exception:
            log.warning("output_bucket_push_failed", path=f"{workspace_id}/{rel}", exc_info=True)


def relocate_outputs_to_run_dir(workspace_id: str, deliverable, out_dir: str):
    """Platform-enforced output organization: move the deliverable's files into
    this run's folder (`out_dir` = `<skill>/<jobshort>`) and return the deliverable
    with paths rewritten to their new locations.

    The skill (an LLM, or a function) can write its outputs ANYWHERE — we never
    rely on it to pick the right folder. At end-of-job the platform files the
    final deliverable under the run folder, so every run's outputs sit in one
    browsable, skill-grouped place no matter what the skill did. Files already
    under `out_dir` (the media default lands them there) are left in place; ones
    written elsewhere (raw bash, an explicit output_path, a `.py` tool) are moved.
    URLs, non-path strings, and paths with no local file pass through unchanged;
    duplicate references to one file resolve to the same new path; basename
    clashes get a `-N` suffix. Walks the deliverable's shape, so nested dict/list
    outputs (e.g. `[{output_url, role}, …]`) are all rewritten.
    """
    from .drive import workspace_drive

    root = workspace_drive(workspace_id).resolve()
    prefix = out_dir.strip("/") + "/"
    moved: dict[str, str] = {}
    used: set[str] = set()

    def _dest_rel(rel: str) -> str:
        base = rel.rsplit("/", 1)[-1]
        cand = f"{out_dir}/{base}"
        if cand not in used and not (root / cand).exists():
            return cand
        stem, dot, ext = base.partition(".")
        n = 1
        while True:
            alt = f"{out_dir}/{stem}-{n}{dot}{ext}"
            if alt not in used and not (root / alt).exists():
                return alt
            n += 1

    def _relocate(value: str) -> str:
        raw = value.strip()
        rel = raw.lstrip("/")
        if not rel or "://" in raw or raw.startswith("data:"):
            return value
        if "/" not in rel or ".." in rel.split("/"):
            return value
        if rel.startswith(prefix):
            return value  # already in the run folder
        if rel in moved:
            return moved[rel]
        src = (root / rel).resolve()
        try:
            src.relative_to(root)
        except ValueError:
            return value
        if not src.is_file():
            # Not local — e.g. produced by a cross-machine subagent (the drive is
            # a per-machine cache over the bucket). Pull on miss before giving
            # up; a path that exists nowhere passes through unchanged.
            if not ensure_local_drive_file(workspace_id, rel):
                return value
        dst_rel = _dest_rel(rel)
        used.add(dst_rel)
        dst = root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src), str(dst))
        except OSError:
            log.warning("output_relocate_failed", frm=rel, to=dst_rel, exc_info=True)
            return value
        moved[rel] = dst_rel
        log.info("output_relocated", frm=rel, to=dst_rel)
        return dst_rel

    def _walk(v):
        if isinstance(v, str):
            return _relocate(v)
        if isinstance(v, dict):
            return {k: _walk(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_walk(x) for x in v]
        return v

    return _walk(deliverable)


def build_output_manifest(workspace_id: str, deliverable) -> list[dict]:
    """The job's deliverable files as `[{name, drive_path, size_bytes, mime}]` for
    `jobs.outputs` — the manifest the dashboard's Outputs view reads to group a
    workspace's results by skill + run without scanning the bucket.

    Walks only the DELIVERABLE (an agent's `set_output` value / a function's
    result), not the whole run record, so step scratch and tool noise stay out.
    Dedups by path and stats each local drive file (outputs are still on local
    disk at end-of-job, before workdir cleanup); a path with no local file (a
    false positive, or an input merely echoed back into the output) is skipped.
    """
    from .drive import workspace_drive

    root = workspace_drive(workspace_id).resolve()
    seen: set[str] = set()
    out: list[dict] = []
    for rel in _output_drive_paths(deliverable):
        if rel in seen:
            continue
        seen.add(rel)
        local = (root / rel).resolve()
        try:
            local.relative_to(root)  # never escape the workspace drive
        except ValueError:
            continue
        if not local.is_file():
            # Cross-machine output (subagent ran elsewhere) — pull on miss so
            # the manifest doesn't silently drop a real deliverable.
            if not ensure_local_drive_file(workspace_id, rel):
                continue
        try:
            size = local.stat().st_size
        except OSError:
            continue
        out.append(
            {
                "name": rel.rsplit("/", 1)[-1],
                "drive_path": rel,
                "size_bytes": size,
                "mime": mimetypes.guess_type(rel)[0] or "application/octet-stream",
            }
        )
    return out


def push_input_files(workspace_id: str, inputs: dict) -> None:
    """Push any worker-local drive files referenced in `inputs` to the bucket.

    A media verb hands the API drive-path inputs (refs, an image to edit); the
    API resolves each to a *signed bucket URL* for Fal, so the bytes must be in
    the bucket first. With FUSE that was implicit; now the worker pushes them.
    Same need when handing a local file to any other out-of-process consumer.
    Best-effort per file — a real miss surfaces downstream as a clean error.
    """
    for rel in _candidate_drive_paths(inputs):
        try:
            upload_drive_file(workspace_id, rel)
        except Exception:
            log.warning("input_bucket_push_failed", path=f"{workspace_id}/{rel}", exc_info=True)


def ensure_local_drive_file(workspace_id: str, rel: str) -> bool:
    """Materialize a single drive object onto the worker's local disk, pulling it
    from the bucket on a local miss. Returns True if the file is present locally
    afterward (already there or freshly pulled), False if it is genuinely not an
    object in the bucket.

    The local drive is a cache over the bucket: a declared input is pulled at
    job start (`ensure_input_files`), but any OTHER bucket object a tool reads —
    a file produced by an earlier job, or one referenced across a process
    boundary — may not be local yet. This is the read-on-miss that fills it: the
    single-path twin of `ensure_input_files`. Used by the tool layer (subagent
    input staging, `file_read`) where we can intercept the read; raw bash can't
    be intercepted, so it pulls explicitly via the `drive_pull` tool.

    Best-effort: a path that isn't a real object just 404s and returns False, so
    the caller still raises a genuine not-found.
    """
    from .drive import workspace_drive

    s = get_settings()
    root = workspace_drive(workspace_id).resolve()
    rel = rel.strip().lstrip("/")
    if rel.startswith("drive/"):
        rel = rel[len("drive/") :]
    dest = (root / rel).resolve()
    try:
        dest.relative_to(root)  # never read/write outside the workspace drive
    except ValueError:
        return False
    if dest.exists():
        return True
    # Offline runner: the local drive is the only store — no bucket to pull from.
    if s.local_mode:
        return False
    try:
        data = download(s.drive_bucket, f"{workspace_id}/{rel}")
    except Exception:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    log.info("drive_file_materialized", path=f"{workspace_id}/{rel}", size=len(data))
    return True


def signed_url(bucket: str, path: str, expires_in: int = 3600) -> str:
    """Time-limited public URL for a private-bucket object. Anyone with the URL
    can fetch it until it expires — fine for handing to an upstream model.

    The target file may only exist on the worker's local disk (it was written
    by bash/python/a tool and never went through an upload path). We push it to
    the bucket first so the URL is actually reachable by Fal. No-op when there's
    nothing local to push (already-remote object).
    """
    # Offline runner: no bucket, so hand back a local file:// URL to the on-disk
    # object (the drive tools that mint URLs are gated off locally, so this is a
    # best-effort fallback rather than a hot path).
    if get_settings().local_mode:
        from .drive import get_drive_root

        return (get_drive_root() / path).resolve().as_uri()
    _ensure_local_object_uploaded(bucket, path)
    res = _client().from_(bucket).create_signed_url(path, expires_in)
    return res["signedURL"]
