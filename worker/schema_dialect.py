"""Puras schema dialect → vanilla JSON Schema translation.

Skill authors write `input_schema` / `output_schema` in a small Puras dialect
that adds end-user-meaningful types on top of JSON Schema:

    type: image | video | audio | file   # accept a file ref (string OR object)
    type: text                            # multi-line string (textarea widget)
    type: color                           # hex string (color-picker widget)

Standard JSON Schema types (`string`, `number`, `integer`, `boolean`, `array`,
`object`, `null`) pass through unchanged. The frontend reads the *dialect*
schema to pick widgets; validators read the *translated* schema (via
`to_jsonschema`) for jsonschema-compatible Draft202012 validation.

This module is duplicated identically into api/app/schema_dialect.py — keep
the two copies in sync.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# Files accept four input shapes (see docs/inputs-and-drive):
#   - bare string  (URL, drive path, or data URI)
#   - { drive_path: "..." }
#   - { url: "..." }
#   - { data: "..." }  (or legacy { base64: "...", media_type: "..." })
_FILE_VALUE_ONEOF: list[dict[str, Any]] = [
    {"type": "string", "minLength": 1},
    {
        "type": "object",
        "additionalProperties": True,
        "anyOf": [
            {"required": ["drive_path"]},
            {"required": ["url"]},
            {"required": ["data"]},
            {"required": ["base64"]},
        ],
    },
]

_FILE_TYPES: frozenset[str] = frozenset({"image", "video", "audio", "file"})

# Same JSON Schema validation, different widget (textarea vs single-line input).
_STRING_ALIAS_TYPES: frozenset[str] = frozenset({"text"})

_COLOR_PATTERN = r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$"

# Standard JSON Schema primitives we pass through.
_STANDARD_TYPES: frozenset[str] = frozenset(
    {"string", "number", "integer", "boolean", "array", "object", "null"}
)

PURAS_TYPES: frozenset[str] = (
    _STANDARD_TYPES | _FILE_TYPES | _STRING_ALIAS_TYPES | {"color"}
)


def to_jsonschema(schema: Any) -> Any:
    """Recursively translate Puras-dialect schema into vanilla JSON Schema.

    Idempotent — schemas that contain only standard JSON Schema pass through
    unchanged. Returns a new dict; the input is not mutated.
    """
    if not isinstance(schema, dict):
        return schema

    out = dict(schema)
    t = out.get("type")

    if isinstance(t, str):
        if t in _FILE_TYPES:
            # Replace the Puras `type: image` with a polymorphic shape.
            # Sibling keys (description, default, examples, title…) stay put
            # — JSON Schema allows them alongside oneOf.
            out.pop("type")
            if "oneOf" not in out and "anyOf" not in out:
                out["oneOf"] = deepcopy(_FILE_VALUE_ONEOF)
        elif t in _STRING_ALIAS_TYPES:
            out["type"] = "string"
        elif t == "color":
            out["type"] = "string"
            out.setdefault("pattern", _COLOR_PATTERN)
        # Anything else (including unknown types) is left untouched — the
        # jsonschema validator will surface the error itself.

    # Recurse into every keyword that contains nested schemas.
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: to_jsonschema(v) for k, v in out["properties"].items()}
    if isinstance(out.get("patternProperties"), dict):
        out["patternProperties"] = {
            k: to_jsonschema(v) for k, v in out["patternProperties"].items()
        }
    ap = out.get("additionalProperties")
    if isinstance(ap, dict):
        out["additionalProperties"] = to_jsonschema(ap)
    items = out.get("items")
    if isinstance(items, dict):
        out["items"] = to_jsonschema(items)
    elif isinstance(items, list):
        out["items"] = [to_jsonschema(s) for s in items]
    if isinstance(out.get("prefixItems"), list):
        out["prefixItems"] = [to_jsonschema(s) for s in out["prefixItems"]]
    for kw in ("oneOf", "anyOf", "allOf"):
        if isinstance(out.get(kw), list):
            out[kw] = [to_jsonschema(s) for s in out[kw]]
    if isinstance(out.get("not"), dict):
        out["not"] = to_jsonschema(out["not"])
    for kw in ("if", "then", "else"):
        if isinstance(out.get(kw), dict):
            out[kw] = to_jsonschema(out[kw])

    return out


def prune_extras(schema: Any, value: Any) -> Any:
    """Return a copy of `value` with object properties not declared in
    `schema.properties` dropped recursively.

    Used to be lenient on tool/skill *outputs*: the agent occasionally tacks
    on a `drive_path` next to an `image`-typed field, or echoes back
    intermediate scratch keys. Silently dropping the extras keeps the run
    succeeding while still flagging *missing* required fields downstream —
    validation runs against the pruned value, so required-field errors and
    type errors still surface.

    Only recurses into `type: object` (with declared `properties`) and
    `type: array` (with a declared `items` schema). All other types pass
    through unchanged — we don't try to peek inside `image`-typed values
    or other dialect leaves.
    """
    if not isinstance(schema, dict):
        return value
    t = schema.get("type")
    if t == "object" and isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        props = schema["properties"]
        return {k: prune_extras(props[k], v) for k, v in value.items() if k in props}
    if t == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        items_schema = schema["items"]
        return [prune_extras(items_schema, v) for v in value]
    return value


def require_all_properties(js: Any) -> Any:
    """Recursively mark every declared property as required — for OUTPUT schemas.

    Output schemas in the Puras dialect don't spell out `required`: the contract
    is that a skill returns *everything* it declares. This walks a *translated*
    JSON Schema (post-`to_jsonschema`) and, for every object node that declares
    `properties` but no explicit `required`, sets `required = list(properties)`.

    An explicit `required` is left untouched, so an author who genuinely needs an
    optional output field can still opt it out by writing the list by hand.

    Pairs with `prune_extras`: undeclared keys are dropped before validation, so
    output schemas need neither `required` nor `additionalProperties`. Returns a
    new dict; the input is not mutated. Mirrors `to_jsonschema`'s recursion.
    """
    if not isinstance(js, dict):
        return js

    out = dict(js)
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: require_all_properties(v) for k, v in props.items()}
        if props and "required" not in out:
            out["required"] = list(props.keys())

    items = out.get("items")
    if isinstance(items, dict):
        out["items"] = require_all_properties(items)
    elif isinstance(items, list):
        out["items"] = [require_all_properties(s) for s in items]
    if isinstance(out.get("prefixItems"), list):
        out["prefixItems"] = [require_all_properties(s) for s in out["prefixItems"]]
    ap = out.get("additionalProperties")
    if isinstance(ap, dict):
        out["additionalProperties"] = require_all_properties(ap)
    for kw in ("oneOf", "anyOf", "allOf"):
        if isinstance(out.get(kw), list):
            out[kw] = [require_all_properties(s) for s in out[kw]]
    if isinstance(out.get("not"), dict):
        out["not"] = require_all_properties(out["not"])
    for kw in ("if", "then", "else"):
        if isinstance(out.get(kw), dict):
            out[kw] = require_all_properties(out[kw])

    return out


def to_output_jsonschema(schema: Any) -> Any:
    """Validator / tool-spec form for OUTPUT schemas: `to_jsonschema` plus
    `require_all_properties`. Inputs keep plain `to_jsonschema` (where an
    explicit `required` lists the genuinely-mandatory fields)."""
    return require_all_properties(to_jsonschema(schema))
