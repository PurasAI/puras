"""Command handlers. Each takes the parsed argparse namespace.

Skillpack resolution order: --skillpack flag → nearest puras.yaml. Auth comes
from `puras login` (~/.puras/config.json) or the PURAS_API_KEY env var.
"""

from __future__ import annotations

import getpass
import io
import json
import re
import sys
import time
import webbrowser
import zipfile
from pathlib import Path

import httpx

from .bundle import zip_skillpack
from .client import ApiClient, ApiError
from .config import (
    DEFAULT_API_BASE,
    PROJECT_FILE,
    Auth,
    clear_auth,
    load_auth,
    load_project,
    save_auth,
    save_project,
)
from .template import scaffold_blank, scaffold_hello_world
from .ui import accent, bold, dim, info, mask_key, ok, table, warn


class CliError(RuntimeError):
    """Expected, user-facing failure — printed without a traceback."""


# ── helpers ──────────────────────────────────────────────────────────────────
def _auth_or_die() -> Auth:
    auth = load_auth()
    if not auth.api_key:
        raise CliError("not logged in — run `puras login`, or set PURAS_API_KEY")
    return auth


def _client() -> ApiClient:
    auth = _auth_or_die()
    return ApiClient(auth.api_base, auth.api_key)


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(s: str) -> bool:
    return bool(_UUID_RE.match((s or "").strip()))


def _skillpack_id(args) -> str:
    sid = getattr(args, "skillpack", None) or load_project().get("skillpack_id")
    if not sid:
        raise CliError(
            "no skillpack — pass --skillpack <id|slug>, name the skill as "
            "`workspace/skillpack/skill`, or run `puras init` in your skillpack dir"
        )
    return sid


def _skill_ref(args) -> tuple[dict, str]:
    """Resolve `puras run <skill>` into (skillpack query params, skill name).

    `<skill>` may be a path copied from a skill's page —
    `workspace/skillpack/skill` or `skillpack/skill` — in which case the
    skillpack comes from the path. A bare skill name falls back to
    `--skillpack` / puras.yaml. A slug ref is sent on `skillpack`; a UUID on
    the legacy `skillpack_id`."""
    raw = (getattr(args, "skill", "") or "").strip().strip("/")
    parts = [p for p in raw.split("/") if p]
    if len(parts) >= 2:
        ref, skill = "/".join(parts[:-1]), parts[-1]
    else:
        ref, skill = _skillpack_id(args), (parts[0] if parts else raw)
    return {"skillpack_id" if _is_uuid(ref) else "skillpack": ref}, skill


def _dashboard_keys_url(api_base: str) -> str:
    if "localhost" in api_base or "127.0.0.1" in api_base:
        return "http://localhost:3000/api-keys"
    return "https://puras.co/api-keys"


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-_")
    s = re.sub(r"-{2,}", "-", s)
    if not s or not s[0].isalnum():
        s = "sp-" + s
    return s[:64]


# ── auth ─────────────────────────────────────────────────────────────────────
def cmd_login(args) -> None:
    current = load_auth()
    api_base = (args.api_base or current.api_base or DEFAULT_API_BASE).rstrip("/")
    key = args.key
    if not key:
        url = _dashboard_keys_url(api_base)
        info(f"Create a workspace API key here: {bold(url)}")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            key = input("Paste your Puras API key: ").strip()
        except EOFError:
            key = ""
    if not key:
        raise CliError("no API key provided")

    client = ApiClient(api_base, key)
    try:
        bal = client.get("/v1/billing/balance")
    except ApiError as e:
        if e.status in (401, 403):
            raise CliError("that API key was rejected — double-check it and try again") from None
        raise
    finally:
        client.close()

    save_auth(api_base, key)
    ok(f"Logged in to {api_base}")
    info(f"  workspace {bal['workspace_id']}  ·  balance ${bal['balance_usd']:.2f}")


def cmd_logout(args) -> None:
    ok("Logged out") if clear_auth() else info("Not logged in")


def cmd_whoami(args) -> None:
    auth = load_auth()
    if not auth.api_key:
        info("not logged in — run `puras login`")
        return
    client = ApiClient(auth.api_base, auth.api_key)
    try:
        bal = client.get("/v1/billing/balance")
    finally:
        client.close()
    info(f"api base   {auth.api_base}")
    info(f"api key    {mask_key(auth.api_key)}")
    info(f"workspace  {bal['workspace_id']}")
    info(f"balance    ${bal['balance_usd']:.2f}")


# ── skillpacks ───────────────────────────────────────────────────────────────
def cmd_init(args) -> None:
    if args.no_scaffold and args.template != "blank":
        raise CliError("--no-scaffold conflicts with --template")
    client = _client()
    name = args.name or Path.cwd().name
    slug = args.slug or _slugify(name)
    body: dict = {"name": name, "slug": slug}
    if args.description:
        body["description"] = args.description
    try:
        sp = client.post("/v1/skillpacks", json_body=body)
    except ApiError as e:
        if e.status == 409:
            raise CliError(f"slug '{slug}' is taken — choose another with --slug") from None
        raise
    finally:
        client.close()

    f = save_project(Path.cwd(), {"skillpack_id": sp["id"], "slug": sp["slug"]})
    # Seed the authored pack-page fields so the file invites editing — the
    # server syncs title/description to the public pack page on deploy.
    seeded = f.read_text()
    if "title:" not in seeded:
        desc = args.description or "What this skillpack does, in one line."
        f.write_text(seeded.rstrip("\n") + f"\n\ntitle: {sp['name']}\ndescription: {desc}\n")
    ok(f"Created skillpack {bold(sp['slug'])}  ({sp['id']})")
    info(f"  wrote {PROJECT_FILE}")
    has_skill = any(
        d.is_dir() and (d / "skill.yaml").is_file() for d in Path.cwd().iterdir()
    )
    if args.no_scaffold:
        info("Next: add a `<skill>/skill.yaml` dir, then `puras deploy`")
    elif has_skill:
        if args.template == "hello-world":
            warn("skills already exist here — skipped the hello-world scaffold")
        else:
            info("Next: add a `<skill>/skill.yaml` dir, then `puras deploy`")
    elif args.template == "hello-world":
        files = scaffold_hello_world(Path.cwd())
        ok(f"Scaffolded the hello-world example ({len(files)} files)")
        info("Next: `puras deploy`, then `puras run greeter -i name=Ada`")
    else:
        scaffold_blank(Path.cwd())
        ok("Scaffolded my-skill/ — edit it, then `puras deploy`")


def cmd_skillpacks(args) -> None:
    client = _client()
    try:
        rows = client.get("/v1/skillpacks")
    finally:
        client.close()
    if not rows:
        info("no skillpacks yet — `puras init`")
        return
    table(
        [[r["slug"], r["name"], "active" if r["active_deployment_id"] else "—", r["id"]] for r in rows],
        ["SLUG", "NAME", "DEPLOY", "ID"],
    )


# ── deployments ──────────────────────────────────────────────────────────────
def _ensure_skillpack(client: ApiClient, args, root: Path) -> str:
    """Resolve the skillpack to deploy to — by id, by slug, or by creating it
    on the first deploy — so you never have to paste a UUID.

    Order:
      1. `--skillpack <uuid>`               → used directly.
      2. puras.yaml `skillpack_id`          → from a prior init/deploy.
      3. slug (`--skillpack <slug>` > puras.yaml `slug` > the dir name) →
         matched against your skillpacks, or created if none matches.

    The resolved id is cached back to `puras.yaml` so later commands (logs,
    secrets, …) find it too."""
    proj = load_project()
    flag = getattr(args, "skillpack", None)
    if flag and _is_uuid(flag):
        return flag
    if not flag and proj.get("skillpack_id"):
        return proj["skillpack_id"]

    slug = flag or proj.get("slug") or _slugify(root.name)
    name = proj.get("title") or root.name
    rows = client.get("/v1/skillpacks") or []
    match = next((r for r in rows if r.get("slug") == slug), None)
    if match:
        sid = match["id"]
    else:
        sp = client.post("/v1/skillpacks", json_body={"name": name, "slug": slug})
        sid = sp["id"]
        slug = sp.get("slug") or slug
        ok(f"Created skillpack {bold(slug)}  ({sid})")
    save_project(root, {"skillpack_id": sid, "slug": slug})
    return sid


def cmd_deploy(args) -> None:
    client = _client()
    try:
        root = Path(args.path or ".").resolve()
        sid = _ensure_skillpack(client, args, root)
        data = zip_skillpack(root)
        info(f"bundling {dim(str(root))}  ({len(data) // 1024 or 1} KiB)")
        form = {"activate": "false" if args.no_activate else "true"}
        if args.notes:
            form["notes"] = args.notes
        dep = client.post(
            f"/v1/skillpacks/{sid}/deployments",
            files={"bundle": ("skillpack.zip", data, "application/zip")},
            data=form,
        )
    finally:
        client.close()
    state = "activated" if dep["is_active"] else "uploaded (not active)"
    skills = len((dep.get("manifest") or {}).get("skills") or [])
    ok(f"Deployed v{dep['version']} · {state}  ({skills} skill{'s' if skills != 1 else ''})")
    if args.no_activate:
        info(f"  activate it with: puras activate {dep['version']}")


def cmd_deployments(args) -> None:
    client = _client()
    try:
        sid = _skillpack_id(args)
        rows = client.get(f"/v1/skillpacks/{sid}/deployments")
    finally:
        client.close()
    if not rows:
        info("no deployments — `puras deploy`")
        return
    table(
        [
            [
                f"v{r['version']}",
                "active" if r["is_active"] else "",
                (r.get("size_bytes") or 0) // 1024,
                str(r["created_at"])[:19].replace("T", " "),
                r["id"],
            ]
            for r in rows
        ],
        ["VERSION", "", "KiB", "CREATED", "ID"],
    )


def cmd_activate(args) -> None:
    client = _client()
    try:
        sid = _skillpack_id(args)
        ref = str(args.ref)
        dep_id = ref
        if ref.isdigit():
            rows = client.get(f"/v1/skillpacks/{sid}/deployments")
            match = [r for r in rows if r["version"] == int(ref)]
            if not match:
                raise CliError(f"no deployment v{ref} in this skillpack")
            dep_id = match[0]["id"]
        res = client.post(f"/v1/skillpacks/{sid}/deployments/{dep_id}/activate")
    finally:
        client.close()
    ok(f"Activated v{res['version']}")


# ── run + logs ───────────────────────────────────────────────────────────────
def _build_inputs(args) -> dict:
    inputs: dict = {}
    if args.json:
        try:
            parsed = json.loads(args.json)
        except ValueError as e:
            raise CliError(f"--json is not valid JSON: {e}") from None
        if not isinstance(parsed, dict):
            raise CliError("--json must be a JSON object")
        inputs.update(parsed)
    for kv in args.input or []:
        key, sep, raw = kv.partition("=")
        if not sep:
            raise CliError(f"--input must be KEY=VALUE (got '{kv}')")
        try:
            inputs[key] = json.loads(raw)  # numbers / bools / objects work
        except ValueError:
            inputs[key] = raw  # plain string
    return inputs


def _run_local(args) -> None:
    """`puras run --local` — drive a local bundle offline on the user's own key.

    The worker runtime ships separately from this SDK (it carries the hosted
    DB/storage stack), so we import it lazily and give a clear message when it
    isn't importable as `worker` (e.g. a standalone `pip install puras`)."""
    try:
        from worker.local_run import LocalRunError, run_local
    except ImportError as e:
        raise CliError(
            "the offline runner isn't installed. Run `pip install puras[local]` "
            "(or `pip install puras-runner`) to add it, then retry "
            f"`puras run --local`. [{e}]"
        ) from None
    inputs = _build_inputs(args)
    bundle_dir = args.dir or "."
    info(f"running `{args.skill or '(sole skill)'}` from {bundle_dir} — offline, your key")
    try:
        res = run_local(
            bundle_dir, inputs,
            skill=args.skill, model=args.model, api_key=args.api_key,
        )
    except LocalRunError as e:
        raise CliError(str(e)) from None
    ok(f"local run · done in {res.get('steps')} steps")
    print(json.dumps(res.get("output"), indent=2, default=str))
    u = res.get("usage") or {}
    info(
        f"  ~{u.get('input_tokens', 0)} in / {u.get('output_tokens', 0)} out tokens "
        f"— billed directly by your LLM provider, not Puras"
    )


def cmd_run(args) -> None:
    if getattr(args, "local", False):
        _run_local(args)
        return
    if not args.skill:
        raise CliError("run requires a skill name (or use --local)")
    client = _client()
    try:
        params, skill = _skill_ref(args)
        inputs = _build_inputs(args)
        if getattr(args, "version", None) is not None:
            params["version"] = str(args.version)
        if not args.async_:
            params["wait"] = "true"
            params["timeout"] = str(args.timeout)
        job = client.post(
            "/v1/jobs",
            params=params,
            json_body={"skill": skill, "inputs": inputs, "source": "cli"},
        )
    finally:
        client.close()

    if args.async_:
        ok(f"Submitted job {job['id']} ({job['status']})")
        info(f"  follow with: puras logs {job['id']}")
        return

    st = job["status"]
    if st == "succeeded":
        ok(f"job {job['id']} · succeeded")
        print(json.dumps(job.get("result"), indent=2))
    elif st in ("failed", "cancelled"):
        warn(f"job {job['id']} · {st}: {job.get('error') or ''}")
        raise SystemExit(1)
    else:
        info(f"job {job['id']} · {st} — still running; `puras logs {job['id']}`")


def cmd_serve(args) -> None:
    """`puras serve` — run a local HTTP API that mirrors the hosted job API,
    backed by the offline runner. Point any Puras SDK at this base URL
    (`apiBase` / `PURAS_API_BASE`) and build/test your whole app offline, on
    your own LLM key, with no puras.co account; flip the base URL to
    api.puras.co + `puras deploy` to ship the identical code."""
    try:
        from worker.local_server import LocalServer
    except ImportError as e:
        raise CliError(
            "the offline runner isn't installed. Run `pip install puras[local]` "
            f"(or `pip install puras-runner`), then retry `puras serve`. [{e}]"
        ) from None

    import os

    root = Path(args.dir or ".").expanduser().resolve()
    if not root.is_dir():
        raise CliError(f"bundle dir not found: {root}")

    app = LocalServer(
        str(root),
        model=args.model,
        api_key=args.api_key,
        require_key=args.require_key,
        on_log=lambda m: print(dim(m)),
    )
    try:
        skills = app.discover_skills()
    except Exception as e:
        raise CliError(f"invalid bundle at {root}: {e}") from None
    if not skills:
        raise CliError(f"no skills found in {root} — need a `<skill>/skill.yaml` dir")

    if not (args.api_key or os.environ.get("ANTHROPIC_API_KEY")):
        warn("no LLM key — set ANTHROPIC_API_KEY (or pass --api-key); jobs will fail until one is set")

    base = f"http://{args.host}:{args.port}"
    ok(f"puras serve · {bold(base)}")
    info(f"  bundle   {dim(str(root))}")
    info(f"  skills   {', '.join(skills)}")
    if args.require_key:
        info(f"  auth     Bearer {mask_key(args.require_key)}")
    else:
        info(dim("  auth     none — pass --require-key to emulate API-key auth"))
    info("")
    info("  Build your app against it — point any Puras SDK at this base URL:")
    info(dim(f'    curl -s "{base}/v1/jobs?wait=true" -H "content-type: application/json" \\'))
    info(dim(f"      -d '{{\"skill\": \"{skills[0]}\", \"inputs\": {{}}}}'"))
    info(dim(f'    Client(api_key="local", api_base="{base}", skillpack="local").run("{skills[0]}", {{}})'))
    info("")
    info(dim("  Ctrl-C to stop"))
    try:
        app.serve_forever(args.host, args.port)
    except KeyboardInterrupt:
        info("\nstopped")


def cmd_replay(args) -> None:
    """Re-run a past job's exact inputs as a new job. Pointed at your LOCAL api
    (with LOCAL_PROJECT_PATH set on the worker) this reproduces the run against
    your local code for debugging — same inputs, your branch."""
    client = _client()
    try:
        params: dict = {}
        if getattr(args, "version", None) is not None:
            params["version"] = str(args.version)
        job = client.post(f"/v1/jobs/{args.job_id}/replay", params=params)
    finally:
        client.close()
    ok(f"Replaying {args.job_id} → new job {job['id']} ({job['status']})")
    where = "local code (LOCAL_PROJECT_PATH)" if job.get("deployment_id") is None else "the active deployment"
    info(f"  running against {where}")
    info(f"  follow with: puras logs {job['id']}")


def _print_event(e: dict) -> None:
    ts = str(e.get("ts", ""))[11:19]
    p = e.get("payload") or {}
    msg = p.get("message") or p.get("text") or p.get("status")
    if msg is None:
        msg = json.dumps(p)[:200] if p else ""
    print(f"{dim(ts)} {accent(str(e.get('type', 'event')))}  {msg}")


def cmd_logs(args) -> None:
    client = _client()
    jid = args.job_id
    seen: set = set()
    terminal = ("succeeded", "failed", "cancelled")
    deadline = time.time() + args.timeout
    try:
        while True:
            for e in client.get(f"/v1/jobs/{jid}/events") or []:
                if e["id"] not in seen:
                    seen.add(e["id"])
                    _print_event(e)
            job = client.get(f"/v1/jobs/{jid}")
            st = job["status"]
            if st in terminal:
                if st == "succeeded":
                    ok(f"job {jid} · succeeded")
                    if job.get("result") is not None:
                        print(json.dumps(job["result"], indent=2))
                else:
                    warn(f"job {jid} · {st}: {job.get('error') or ''}")
                    raise SystemExit(1)
                return
            if time.time() > deadline:
                info(f"(timeout) job is still {st} — `puras logs {jid}` to keep watching")
                return
            time.sleep(args.interval)
    finally:
        client.close()


def cmd_spans(args) -> None:
    """Render a job's trace spans (P0-3) as an indented latency waterfall, so you
    can see which step / model call / tool / subagent dominated the run."""
    client = _client()
    try:
        spans = client.get(f"/v1/jobs/{args.job_id}/spans") or []
    finally:
        client.close()
    if args.json:
        print(json.dumps(spans, indent=2))
        return
    if not spans:
        info("no spans recorded for this job (older job, or it failed early)")
        return
    # Build the parent→children tree and print it depth-first in record order.
    children: dict = {}
    roots = []
    for s in spans:
        children.setdefault(s["parent_span_id"], []).append(s)
        if s["parent_span_id"] is None:
            roots.append(s)

    def _walk(span, depth):
        attrs = span.get("attributes") or {}
        tag = attrs.get("tool") or attrs.get("model") or ""
        extra = f" {tag}" if tag else ""
        if attrs.get("ok") is False:
            extra += " [err]"
        print(f"{'  ' * depth}{span['kind']:<7} {span['duration_ms']:>7}ms  {span['name']}{extra}")
        for ch in children.get(span["span_id"], []):
            _walk(ch, depth + 1)

    for r in roots:
        _walk(r, 0)


# ── secrets ──────────────────────────────────────────────────────────────────
def cmd_secret_set(args) -> None:
    name, sep, value = args.assignment.partition("=")
    if not sep:
        value = getpass.getpass(f"Value for {name}: ")
    client = _client()
    try:
        sid = _skillpack_id(args)
        client.put(f"/v1/skillpacks/{sid}/secrets", json_body={"name": name, "value": value})
    finally:
        client.close()
    ok(f"Set secret {name}")


def cmd_secret_ls(args) -> None:
    client = _client()
    try:
        sid = _skillpack_id(args)
        rows = client.get(f"/v1/skillpacks/{sid}/secrets")
    finally:
        client.close()
    if not rows:
        info("no secrets set")
        return
    table(
        [[r["name"], str(r["updated_at"])[:19].replace("T", " ")] for r in rows],
        ["NAME", "UPDATED"],
    )


def cmd_secret_rm(args) -> None:
    client = _client()
    try:
        sid = _skillpack_id(args)
        client.delete(f"/v1/skillpacks/{sid}/secrets/{args.name}")
    finally:
        client.close()
    ok(f"Deleted secret {args.name}")


# ── pull ─────────────────────────────────────────────────────────────────────
def cmd_pull(args) -> None:
    client = _client()
    try:
        sid = _skillpack_id(args)
        res = client.get(f"/v1/skillpacks/{sid}/deployments/pull", params={"ttl": "600"})
        data = httpx.get(res["download_url"], follow_redirects=True, timeout=120.0).content
    finally:
        client.close()

    out = Path(args.out or ".").resolve()
    zf = zipfile.ZipFile(io.BytesIO(data))
    for n in zf.namelist():
        if n.startswith("/") or ".." in Path(n).parts:
            raise CliError(f"refusing to extract unsafe path from bundle: {n}")
    out.mkdir(parents=True, exist_ok=True)
    zf.extractall(out)
    ok(f"Pulled v{res['version']} → {out}")


# ── evals ──────────────────────────────────────────────────────────────────
def _usd(micros) -> str:
    if micros is None:
        return "—"
    return f"${micros / 1_000_000:.4f}"


def _print_eval_report(rep: dict) -> None:
    """One-screen summary of an eval suite report."""
    st = rep["status"]
    badge = green(st) if st == "succeeded" else red(st) if st in ("failed", "cancelled") else accent(st)
    info("")
    info(f"{bold('Eval suite')} {rep['id']}  ·  {bold(rep['skill_name'])}"
         + (f" @v{rep['version']}" if rep.get("version") else "")
         + f"  ·  {badge}")
    pr = rep.get("pass_rate_pct")
    info(
        f"  pass-rate: {bold(f'{pr}%' if pr is not None else '—')}"
        f"   ·   mean score: {rep.get('mean_score') if rep.get('mean_score') is not None else '—'}"
        f" ± {rep.get('score_stddev') if rep.get('score_stddev') is not None else '—'}"
        f"   ·   {rep['completed_runs']}/{rep['total_runs']} runs"
    )
    info(
        f"  cost: {_usd(rep.get('total_cost_micros'))} total"
        f" ({_usd(rep.get('mean_cost_micros'))}/run)"
        f"   ·   latency: {rep.get('mean_latency_ms') if rep.get('mean_latency_ms') is not None else '—'} ms/run"
        + (f"   ·   threshold: {rep['threshold']}%" if rep.get("threshold") is not None else "")
    )
    graders = rep.get("graders") or []
    if graders:
        table(
            [
                [
                    g["name"],
                    g.get("kind") or "—",
                    f"{round(g['mean_score'] * 100)}%" if g.get("mean_score") is not None else "—",
                    f"{round(g['pass_rate'] * 100)}%" if g.get("pass_rate") is not None else "—",
                    g.get("runs", 0),
                ]
                for g in graders
            ],
            ["grader", "kind", "mean", "pass", "runs"],
        )


def _eval_local(args) -> None:
    """`puras eval --local` — run a skill's eval suite offline on the local runner."""
    try:
        from worker.eval_local import run_eval_local
        from worker.local_run import LocalRunError
    except ImportError as e:
        raise CliError(
            "the offline runner isn't installed. Run `pip install puras[local]` "
            f"(or `pip install puras-runner`), then retry `puras eval --local`. [{e}]"
        ) from None
    bundle_dir = args.dir or "."
    info(f"evaluating `{args.skill or '(sole skill)'}` from {bundle_dir} — offline, your key")
    try:
        rep = run_eval_local(
            bundle_dir,
            skill=args.skill, model=args.model, api_key=args.api_key,
            case_ids=args.case, repeat=args.repeat or 1, threshold=args.threshold,
        )
    except LocalRunError as e:
        raise CliError(str(e)) from None
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        gate = "" if rep["threshold"] is None else (
            f"  (gate {rep['pass_rate_pct']}% {'≥' if rep['gate_passed'] else '<'} "
            f"{rep['threshold']}%)"
        )
        ok(
            f"local eval · {rep['passed']}/{rep['total']} passed · "
            f"pass-rate {rep['pass_rate_pct']}% · mean score {rep['mean_score']}{gate}"
        )
    if rep.get("gate_passed") is False:
        raise SystemExit(1)


def cmd_eval(args) -> None:
    if getattr(args, "local", False):
        _eval_local(args)
        return
    if not args.skill:
        raise CliError("eval requires a skill name (or use --local)")
    client = _client()
    suite_id = None
    try:
        sid = _skillpack_id(args)
        body: dict = {"skill": args.skill}
        if getattr(args, "version", None) is not None:
            body["version"] = args.version
        if getattr(args, "repeat", None):
            body["repeat"] = args.repeat
        if getattr(args, "threshold", None) is not None:
            body["threshold"] = args.threshold
        if getattr(args, "case", None):
            body["case_ids"] = args.case
        rep = client.post(f"/v1/skillpacks/{sid}/evals", json_body=body)
        suite_id = rep["id"]
        ok(f"Started eval suite {suite_id} — {rep['total_runs']} run(s)")
        if args.async_:
            info(f"  follow with: puras eval-report {suite_id}")
            return
        # Poll to completion.
        terminal = ("succeeded", "failed", "cancelled")
        deadline = time.time() + args.timeout
        last = -1
        while True:
            rep = client.get(f"/v1/evals/{suite_id}")
            done = rep["completed_runs"]
            if done != last:
                info(dim(f"  … {done}/{rep['total_runs']} runs complete"))
                last = done
            if rep["status"] in terminal:
                break
            if time.time() > deadline:
                info(f"(timeout) suite still running — `puras eval-report {suite_id}`")
                return
            time.sleep(args.interval)
    finally:
        client.close()

    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        _print_eval_report(rep)
    # CI gate: a non-succeeded suite (below threshold, errored, or cancelled)
    # exits non-zero so a pipeline step fails on regression.
    if rep.get("gate_passed") is False:
        raise SystemExit(1)


def cmd_eval_report(args) -> None:
    client = _client()
    try:
        rep = client.get(f"/v1/evals/{args.suite_id}")
    finally:
        client.close()
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        _print_eval_report(rep)
    if rep.get("gate_passed") is False:
        raise SystemExit(1)


def cmd_eval_diff(args) -> None:
    client = _client()
    try:
        sid = _skillpack_id(args)
        diff = client.get(
            f"/v1/skillpacks/{sid}/evals/diff",
            params={"skill": args.skill, "base": args.base, "head": args.head},
        )
    finally:
        client.close()
    if args.json:
        print(json.dumps(diff, indent=2))
        if diff.get("regressed"):
            raise SystemExit(1)
        return
    b, h, d = diff["base"], diff["head"], diff["deltas"]
    info(f"{bold('Eval diff')} {diff['skill']}  ·  base v{b.get('version')} → head v{h.get('version')}")
    table(
        [
            ["pass-rate %", b.get("pass_rate_pct"), h.get("pass_rate_pct"), d.get("pass_rate_pct")],
            ["mean score", b.get("mean_score"), h.get("mean_score"), d.get("mean_score")],
            ["mean cost", _usd(b.get("mean_cost_micros")), _usd(h.get("mean_cost_micros")),
             (f"{d['mean_cost_micros'] / 1_000_000:+.4f}" if d.get("mean_cost_micros") is not None else "—")],
            ["mean latency ms", b.get("mean_latency_ms"), h.get("mean_latency_ms"), d.get("mean_latency_ms")],
        ],
        ["metric", "base", "head", "Δ"],
    )
    if diff.get("regressed"):
        warn("regression: head pass-rate is below base")
        raise SystemExit(1)
    ok("no pass-rate regression")
