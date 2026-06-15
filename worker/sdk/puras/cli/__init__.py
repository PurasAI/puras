"""`puras` CLI entrypoint — argparse wiring + dispatch.

Console script: `puras = puras.cli:main` (see pyproject). Also runnable as
`python -m puras` / `python -m puras.cli`.
"""

from __future__ import annotations

import argparse

from . import commands as c
from .client import ApiError
from .commands import CliError
from .ui import warn


def _version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("puras")
    except Exception:
        try:
            from .. import __version__

            return __version__
        except Exception:
            return "0+local"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="puras",
        description="Deploy and call long-running agentic skills on Puras.",
    )
    p.add_argument("--version", action="version", version=f"puras {_version()}")
    sub = p.add_subparsers(dest="cmd", metavar="<command>")

    def add(name: str, fn, help: str, aliases: tuple[str, ...] = ()) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help, aliases=list(aliases))
        sp.set_defaults(func=fn)
        return sp

    lp = add("login", c.cmd_login, "store + verify a workspace API key")
    lp.add_argument("--key", help="API key (else you'll be prompted / browser opens)")
    lp.add_argument("--api-base", help=f"override API base URL")

    add("logout", c.cmd_logout, "remove stored credentials")
    add("whoami", c.cmd_whoami, "show the active workspace + balance")

    ip = add("init", c.cmd_init, "scaffold a skill and write puras.yaml")
    ip.add_argument("--name")
    ip.add_argument("--slug")
    ip.add_argument("--description")
    ip.add_argument(
        "--template",
        choices=["blank", "hello-world"],
        default="blank",
        help="starter files: a minimal blank skill (default) or the full "
        "hello-world example (github.com/PurasAI/hello-world)",
    )
    ip.add_argument("--no-scaffold", action="store_true", help="don't write any starter files")

    add("skillpacks", c.cmd_skillpacks, "list your skills", aliases=("skills",))

    dp = add("deploy", c.cmd_deploy, "bundle a skill dir and push a deployment")
    dp.add_argument("path", nargs="?", help="skill dir (default: .)")
    dp.add_argument(
        "--app", "--skillpack", dest="skillpack",
        help="skill id or slug to deploy to (default: puras.yaml, the skill's "
        "name, or the dir name; created on first deploy)",
    )
    dp.add_argument("--no-activate", action="store_true", help="upload without activating")
    dp.add_argument("--notes")

    lsp = add("deployments", c.cmd_deployments, "list deployments for the skill")
    lsp.add_argument("--app", "--skillpack", dest="skillpack")

    ap = add("activate", c.cmd_activate, "activate a deployment by version or id")
    ap.add_argument("ref", help="version number (e.g. 3) or deployment id")
    ap.add_argument("--app", "--skillpack", dest="skillpack")

    rp = add("run", c.cmd_run, "submit a job and wait for the result")
    rp.add_argument(
        "skill", nargs="?",
        help="skill name, or a path copied from the skill's page "
        "(optional with --local when the bundle has one skill)",
    )
    rp.add_argument(
        "-p", "--prompt",
        help="run a one-off agent from this inline prompt — NO skill, NO deploy "
        "needed. Use `-` to read stdin or `@file` to read a file.",
    )
    rp.add_argument("-i", "--input", action="append", metavar="KEY=VALUE", help="repeatable")
    rp.add_argument("--json", help="inputs as a JSON object")
    rp.add_argument("--async", dest="async_", action="store_true", help="don't wait")
    rp.add_argument("--timeout", type=int, default=60, help="wait seconds (default 60)")
    rp.add_argument("--app", "--skillpack", dest="skillpack", help="default skill id or slug for a bare skill name")
    rp.add_argument("--version", type=int, help="pin to a deployment version (default: active)")
    # Offline mode: run a local bundle with no platform, on your own LLM key.
    rp.add_argument(
        "--local", action="store_true",
        help="run a local skill bundle OFFLINE (no platform) on your own LLM key",
    )
    rp.add_argument(
        "--dir", help="bundle dir for --local (default: current directory)"
    )
    rp.add_argument(
        "--api-key", dest="api_key",
        help="LLM key for --local (default: $ANTHROPIC_API_KEY)",
    )
    rp.add_argument("--model", help="override the skill's model slug for --local")

    # Local API server: mirror the hosted job API on localhost so an app can be
    # built/tested against it offline, on the user's own LLM key, no account.
    svp = add("serve", c.cmd_serve, "serve a local API for a bundle (the hosted job API, offline)")
    svp.add_argument("--dir", help="bundle dir to serve (default: current directory)")
    svp.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    svp.add_argument("--port", type=int, default=8787, help="bind port (default 8787)")
    svp.add_argument(
        "--api-key", dest="api_key",
        help="LLM key used to run jobs (default: $ANTHROPIC_API_KEY)",
    )
    svp.add_argument("--model", help="override each skill's model slug")
    svp.add_argument(
        "--require-key", dest="require_key", metavar="TOKEN",
        help="emulate API-key auth: require `Authorization: Bearer TOKEN` on /v1/* requests",
    )

    gp = add("logs", c.cmd_logs, "stream a job's events until it finishes")
    gp.add_argument("job_id")
    gp.add_argument("--timeout", type=int, default=600)
    gp.add_argument("--interval", type=float, default=1.5)

    sp = add("spans", c.cmd_spans, "show a job's trace spans (run/step/model/tool latency)")
    sp.add_argument("job_id")
    sp.add_argument("--json", action="store_true", help="print the raw spans as JSON")

    rpl = add("replay", c.cmd_replay, "re-run a past job's inputs (reproduce it locally)")
    rpl.add_argument("job_id", help="the job to replay")
    rpl.add_argument("--version", type=int, help="pin a deployment version (default: active / local)")

    ep = add("eval", c.cmd_eval, "run a skill's eval suite (dataset + graders)")
    ep.add_argument(
        "skill", nargs="?",
        help="skill name (resolved from --app / puras.yaml; "
        "optional with --local when the bundle has one skill)",
    )
    ep.add_argument("--app", "--skillpack", dest="skillpack", help="skill id (default: puras.yaml binding)")
    ep.add_argument("--version", type=int, help="pin to a deployment version (default: active)")
    ep.add_argument("--repeat", type=int, default=1, help="run each case N times for variance (default 1)")
    ep.add_argument("--threshold", type=int, help="CI gate: fail if pass-rate %% is below this")
    ep.add_argument("--case", action="append", help="run only this case id (repeatable)")
    ep.add_argument("--json", action="store_true", help="print the full report as JSON")
    ep.add_argument("--async", dest="async_", action="store_true", help="kick off and return; don't wait")
    ep.add_argument("--timeout", type=int, default=900, help="max seconds to wait (default 900)")
    ep.add_argument("--interval", type=float, default=3.0, help="poll interval seconds (default 3)")
    # Offline mode: run the suite locally with no platform, on your own LLM key.
    ep.add_argument(
        "--local", action="store_true",
        help="run the eval suite OFFLINE on a local bundle, on your own LLM key",
    )
    ep.add_argument("--dir", help="bundle dir for --local (default: current directory)")
    ep.add_argument(
        "--api-key", dest="api_key",
        help="LLM key for --local (default: $ANTHROPIC_API_KEY)",
    )
    ep.add_argument("--model", help="override the skill's model slug for --local")

    erp = add("eval-report", c.cmd_eval_report, "show an eval suite report by id")
    erp.add_argument("suite_id")
    erp.add_argument("--json", action="store_true", help="print the full report as JSON")

    edp = add("eval-diff", c.cmd_eval_diff, "A/B two eval suites (version vs version)")
    edp.add_argument("skill", help="the skill both suites evaluated")
    edp.add_argument("--app", "--skillpack", dest="skillpack", help="skill id (default: puras.yaml binding)")
    edp.add_argument("--base", required=True, help="baseline: a version number or a suite id")
    edp.add_argument("--head", required=True, help="candidate: a version number or a suite id")
    edp.add_argument("--json", action="store_true", help="print the full diff as JSON")

    secp = sub.add_parser("secrets", help="manage your skill's secrets")
    secp.set_defaults(func=lambda _a: secp.print_help())
    secsub = secp.add_subparsers(dest="subcmd", metavar="<set|ls|rm>")
    ss = secsub.add_parser("set", help="set a secret (NAME=VALUE or prompt)")
    ss.add_argument("assignment", metavar="NAME[=VALUE]")
    ss.add_argument("--app", "--skillpack", dest="skillpack")
    ss.set_defaults(func=c.cmd_secret_set)
    sl = secsub.add_parser("ls", help="list secret names")
    sl.add_argument("--app", "--skillpack", dest="skillpack")
    sl.set_defaults(func=c.cmd_secret_ls)
    sr = secsub.add_parser("rm", help="delete a secret")
    sr.add_argument("name")
    sr.add_argument("--app", "--skillpack", dest="skillpack")
    sr.set_defaults(func=c.cmd_secret_rm)

    pp = add("pull", c.cmd_pull, "download the active bundle")
    pp.add_argument("--app", "--skillpack", dest="skillpack")
    pp.add_argument("--out", help="output dir (default: .)")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    try:
        func(args)
        return 0
    except CliError as e:
        warn(str(e))
        return 1
    except ApiError as e:
        warn(f"API error {e.status}: {e.detail}")
        return 1
    except KeyboardInterrupt:
        return 130
