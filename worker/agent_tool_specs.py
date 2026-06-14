"""Anthropic tool specs for the built-in agent tools.

Kept in its own module (zero runtime deps) so the docs generator script
can import them without dragging in the worker's db / provider chain.
The agent loop imports the same constants from here.
"""

BASH_TOOL_SPEC = {
    "name": "bash",
    "description": (
        "Run a shell command in the job's working directory. The current dir "
        "contains a `drive/` folder for files that should persist across jobs "
        "(synced to the workspace drive). Anything written elsewhere is "
        "ephemeral. Returns combined stdout+stderr (last 8KB) and the exit code."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {
                "type": "integer",
                "description": "Max seconds (default 60, hard ceiling 600)",
            },
        },
        "required": ["command"],
    },
}

TODO_WRITE_TOOL_SPEC = {
    "name": "todo_write",
    "description": (
        "Record or update your working plan as an ordered checklist. Pass the "
        "FULL list every time — it REPLACES the previous one (this tool keeps no "
        "state of its own). Use it to break a non-trivial job into concrete "
        "steps before you start, then keep it live: mark a step `in_progress` "
        "right before you begin it and `completed` the moment it's done. Keep "
        "exactly one step `in_progress` at a time. The list is surfaced to the "
        "user as a live progress checklist; it does not run anything itself. "
        "Skip it for trivial one-step tasks — reach for it when work spans "
        "several stages (e.g. spec → plan → build → validate)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": (
                    "The complete, ordered todo list. Replaces any prior list, "
                    "so always send every item, not just the changed ones."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": (
                                "Imperative, specific description of the step "
                                "(e.g. 'Implement the virtual joystick controller')."
                            ),
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Current state of this step.",
                        },
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["todos"],
    },
}

_VERB_AUTO_NOTE = (
    "You don't pick the model — it's selected automatically from the inputs you "
    "pass (which fix the task) and the skill's configured default, overridable per "
    "run. Express what you need through the inputs, not a model name."
)

_VERB_URL_NOTE = (
    "Any image/video/audio reference can be a drive path (e.g. an input file, or "
    "the `drive_path` a previous step returned) OR an https/data URL — the "
    "platform resolves drive paths to fetchable URLs (and resizes images to a "
    "model's minimum) automatically, so pass the path straight in; there's no "
    "need to call `drive_url` first."
)

# How to point at a specific reference image from inside the prompt. ONE portable
# convention — the platform normalizes it per model, so the same prompt works
# whichever model the user picks.
_VERB_REF_TAG_NOTE = (
    "To bind part of the prompt to a specific reference, write `@Image1`, "
    "`@Image2`, … (1-indexed, in the SAME order as `refs`) — e.g. \"@Image1 is the "
    "product the person in @Image2 holds\". This is portable: models that read the "
    "tags natively (Seedance and Kling reference-to-video) get them verbatim, and "
    "every other model (Veo, and all image models) has them rewritten to prose "
    "(\"the first reference image\") automatically — so write `@ImageN` once and it "
    "works on whichever model is selected. Don't use `@ElementN` (not wired) — "
    "always `@ImageN`."
)

GENERATE_IMAGE_TOOL_SPEC = {
    "name": "generate_image",
    "description": (
        "Generate an image from a prompt. Pass `refs` (drive paths or image "
        "URLs) to edit/compose from references instead of plain text-to-image. "
        "Prefer this over the raw `media` tool. Returns the drive `drive_path` "
        "where the file was saved — pass that path straight into another "
        "`generate_*` call (as `refs`/`image`) without converting it. "
        "Give EXACTLY ONE of `prompt` (inline) or `prompt_path` (a drive text "
        "file you wrote with `file_write`). "
        + _VERB_AUTO_NOTE + " " + _VERB_URL_NOTE + " " + _VERB_REF_TAG_NOTE
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What to draw, or how to edit the refs. HARD LIMIT ~2500 characters: an over-limit prompt is REJECTED with an error before anything is rendered or billed — it is NOT truncated, so keep the prompt under ~2500 (aim under ~1900) and put the must-render elements first. Point at a specific reference with `@Image1`/`@Image2` (in `refs` order). For a long prompt, prefer `prompt_path` so you don't re-emit the text here."},
            "prompt_path": {"type": "string", "description": "Drive path of a UTF-8 text file whose contents become the prompt (exactly one of `prompt`/`prompt_path`). Token-saver for long prompts: `file_write` the prompt once — the result reports its exact `chars`, so you verify the budget for free — then pass the path here instead of pasting the text again. Trim with `file_edit` if over budget. The same char cap applies to the file's contents."},
            "refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Reference images (drive paths or URLs) → edit/compose mode. Omit for text-to-image. Address them in the prompt as `@Image1`, `@Image2`, … in this order.",
            },
            "aspect_ratio": {"type": "string", "description": "e.g. '1:1', '16:9', '9:16'."},
            "resolution": {"type": "string", "description": "'1K' | '2K' | '4K' (where supported)."},
            "n": {"type": "integer", "description": "Number of images (default 1)."},
            "output_path": {"type": "string", "description": "Optional drive subpath (e.g. 'shots/hero.png')."},
        },
        "required": [],
    },
}

GENERATE_VIDEO_TOOL_SPEC = {
    "name": "generate_video",
    "description": (
        "Generate a video. The inputs you pass fix the task: `lipsync_audio` "
        "(+`image`) → lip-synced talking head; `refs` → reference-to-video; "
        "`image` → image-to-video; neither → text-to-video. Prefer this over "
        "the raw `media` tool. Returns the drive `drive_path`. "
        + _VERB_AUTO_NOTE + " " + _VERB_URL_NOTE + " " + _VERB_REF_TAG_NOTE
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The scene/action (a short delivery note for lip-sync). "
                    "Multi-shot: label beats `Shot 1:` / `Cut to:` — understood "
                    "across models; Seedance also follows timed cues like `Cut 1 "
                    "(0.0-0.7s) …`. Spoken dialogue is generated NATIVELY when "
                    "`audio: true` — write it as `Speaker says: \"<line>\"` (a "
                    "colon, NOT quotes around the speaker) and append `(no "
                    "subtitles, no on-screen text)`, since models trained on "
                    "subtitled clips otherwise stamp garbled captions. Keep on-"
                    "screen words OUT of the prompt (models garble >2-3 rendered "
                    "words) — render copy as a caption/still afterward instead. "
                    "Reference a specific image with `@Image1`/`@Image2`. For a "
                    "long prompt, prefer `prompt_path`."
                ),
            },
            "prompt_path": {
                "type": "string",
                "description": (
                    "Drive path of a UTF-8 text file whose contents become the "
                    "prompt (give at most one of `prompt`/`prompt_path`). "
                    "Token-saver for long prompts: `file_write` the prompt once "
                    "(its result reports exact `chars`), then pass the path "
                    "here instead of pasting the text again."
                ),
            },
            "image": {"type": "string", "description": "A still to animate (i2v) or the portrait (lip-sync). Drive path or URL."},
            "last_frame": {
                "type": "string",
                "description": (
                    "Optional still (drive path or URL) the clip should END on "
                    "(tail keyframe). Pair with `image` for a first→last keyframe "
                    "interpolation, or use alone to land a t2v/r2v clip on a set "
                    "frame (e.g. an end card, or a gameplay first frame so the clip "
                    "cuts cleanly into real footage). Seedance / Kling only; "
                    "ignored with a warning on other families and on lip-sync."
                ),
            },
            "refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Reference images (drive paths or URLs) → reference-to-video (keep referenced subjects on-model). Address them in the prompt as `@Image1`, `@Image2`, … in this order.",
            },
            "lipsync_audio": {"type": "string", "description": "Narration audio (drive path or URL) → lip-sync mode (talking head). Only for a verbatim read; for a creator/skit speaking, put the line in the prompt with `audio: true` instead."},
            "duration": {
                "type": "integer",
                "description": "Seconds. Snapped to the model's allowed set (e.g. Veo 4/6/8) with a warning. Ignored for lip-sync.",
            },
            "aspect_ratio": {"type": "string", "description": "e.g. '16:9', '9:16', '1:1'."},
            "resolution": {"type": "string", "description": "e.g. '720p', '1080p' (where supported)."},
            "audio": {"type": "boolean", "description": "Generate native audio (t2v/i2v/r2v) — dialogue, SFX and ambience in the same pass. Every family (Seedance / Kling / Veo) supports it; put any spoken line in the prompt (`Speaker says: \"…\"`). Default false."},
            "output_path": {"type": "string", "description": "Optional drive subpath (e.g. 'renders/clip.mp4')."},
        },
        "required": [],
    },
}

GENERATE_AUDIO_TOOL_SPEC = {
    "name": "generate_audio",
    "description": (
        "Generate speech from text (text-to-speech). Prefer this over the raw "
        "`media` tool. Returns the drive `drive_path` of the audio file. The "
        "default model is ElevenLabs v3, which reads inline AUDIO TAGS — wrap a "
        "delivery cue in square brackets and v3 acts it out without speaking it: "
        "'[excited] It's finally here! [whispers] Don't tell anyone.' Common "
        "tags: [excited] [sad] [angry] [whispers] [laughs] [sighs] [sarcastic] "
        "[British accent]. (To transcribe speech→text, use the raw `media` tool "
        "with a speech-to-text model instead.) " + _VERB_AUTO_NOTE
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The words to speak (billed per character). May contain v3 audio tags in [brackets]."},
            "voice": {"type": "string", "description": "A voice name/persona (e.g. 'Aria', 'Roger') or a voice id."},
            "language": {"type": "string", "description": "ISO 639-1 code (e.g. 'en', 'tr'); omit to auto-detect."},
            "stability": {"type": "number", "description": "Delivery (0–1): lower = more expressive and tag-responsive, higher = steadier. The primary v3 knob."},
            "similarity_boost": {"type": "number", "description": "Delivery: similarity boost. v2-only (ignored on the default v3 model)."},
            "style": {"type": "number", "description": "Delivery: style exaggeration. v2-only (ignored on the default v3 model)."},
            "speed": {"type": "number", "description": "Delivery: speaking speed. v2-only (ignored on the default v3 model)."},
            "output_path": {"type": "string", "description": "Optional drive subpath (e.g. 'audio/vo.mp3')."},
        },
        "required": ["text"],
    },
}

TRANSCRIBE_TOOL_SPEC = {
    "name": "transcribe",
    "description": (
        "Transcribe speech to text. Pass `audio` (a drive path, or an https/data "
        "URL); returns the transcript inline as `text` plus `words` (each with "
        "`start`/`end` seconds) — there is NO file. Optionally pass `keyterms` "
        "(brand / proper-noun spellings) to bias the transcription, and a "
        "`language` hint. The platform resolves a drive path to a fetchable URL "
        "automatically. (To go the other way — text → speech — use "
        "`generate_audio`.)"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "audio": {
                "type": "string",
                "description": "Audio to transcribe — a drive path or an https/data URL.",
            },
            "keyterms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional brand / proper-noun spellings to bias transcription.",
            },
            "language": {
                "type": "string",
                "description": "Optional ISO 639-1 hint (e.g. 'en', 'tr'); omit to auto-detect.",
            },
        },
        "required": ["audio"],
    },
}

WEB_SEARCH_TOOL_SPEC = {
    "name": "web_search",
    "description": (
        "Search the web via the platform's search provider. Returns a list of "
        "results with title, url, and a short snippet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Max results to return (1-20, default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

IMAGE_SEARCH_TOOL_SPEC = {
    "name": "image_search",
    "description": (
        "Search for images on the web via the platform's search provider. "
        "Returns image URLs, thumbnails, dimensions, and source pages."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Image search query"},
            "max_results": {
                "type": "integer",
                "description": "Max results to return (1-20, default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

WEB_FETCH_TOOL_SPEC = {
    "name": "web_fetch",
    "description": (
        "Fetch a web page (HTTP GET) and return its plain text content with "
        "scripts/styles stripped. By default JavaScript is NOT executed, so a "
        "client-rendered SPA comes back mostly empty — set `render_js: true` to "
        "render the page in a headless browser first and return the text the "
        "user would actually see (slower, but works for JS apps). Returns the "
        "final URL (after redirects), page title, and extracted text (truncated "
        "to `max_chars`)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch (http:// or https://)"},
            "max_chars": {
                "type": "integer",
                "description": "Max chars of body text to return (500-200000, default 20000)",
                "default": 20000,
            },
            "render_js": {
                "type": "boolean",
                "description": (
                    "Render the page in a headless browser (Chromium) so its "
                    "JavaScript runs before the text is extracted. Use for "
                    "single-page apps / sites that render client-side. Default "
                    "false (plain HTTP fetch)."
                ),
                "default": False,
            },
        },
        "required": ["url"],
    },
}

WEB_SCREENSHOT_TOOL_SPEC = {
    "name": "web_screenshot",
    "description": (
        "Render a page in a headless browser (Chromium) and capture a PNG "
        "screenshot, saved to the workspace drive. Point it at EITHER a live "
        "`url` OR a `path` to an HTML file already in the drive (e.g. a "
        "single-file playable you just built) — exactly one. JavaScript runs "
        "before the shot, so this is how you see what a client-rendered page or "
        "an HTML5 game actually looks like. The screenshot is also attached to "
        "the conversation so you can inspect it directly (on vision models), and "
        "any console/page errors are reported — useful for validating that a "
        "built HTML document renders without crashing. Returns the saved "
        "`drive_path` (sign it with `drive_url` only if you need a URL), the "
        "page title, and any `console_errors`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "A live page to render (http:// or https://). Mutually "
                    "exclusive with `path`."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Drive path to a local HTML file to render (e.g. "
                    "'_jobs/<job_id>/playable.html'). A leading 'drive/' is "
                    "accepted and stripped. Mutually exclusive with `url`."
                ),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Drive path to save the PNG to (e.g. "
                    "'_jobs/<job_id>/preview.png'). Defaults to a generated path "
                    "under 'screenshots/'. Must end in '.png'."
                ),
            },
            "full_page": {
                "type": "boolean",
                "description": (
                    "Capture the full scrollable page instead of just the "
                    "viewport. Default false. Avoid it on LONG pages: a tall "
                    "capture can exceed the model's per-side pixel limit, in "
                    "which case the image is saved to the drive but NOT attached "
                    "for you to view. To inspect a long page, leave this false "
                    "and use `scroll_y` to grab specific sections as single "
                    "viewport shots."
                ),
                "default": False,
            },
            "viewport_width": {
                "type": "integer",
                "description": "Viewport width in px (64-3840, default 1280).",
                "default": 1280,
            },
            "viewport_height": {
                "type": "integer",
                "description": "Viewport height in px (64-3840, default 800).",
                "default": 800,
            },
            "scroll_y": {
                "type": "integer",
                "description": (
                    "Scroll the page down this many pixels before capturing a "
                    "single viewport (default 0 = top). Use this to inspect a "
                    "section further down a long page without a giant full_page "
                    "capture — e.g. scroll_y: 1400 to see the next screenful."
                ),
                "default": 0,
            },
            "wait_ms": {
                "type": "integer",
                "description": (
                    "Extra milliseconds to wait after load before capturing, so "
                    "animations/first frame settle (0-15000, default 1200)."
                ),
                "default": 1200,
            },
        },
    },
}

FILE_READ_TOOL_SPEC = {
    "name": "file_read",
    "description": (
        "Read one or more files from the workspace drive and attach them to "
        "the conversation. Images (jpg/png/gif/webp) and PDFs come back as "
        "vision/document blocks you can look at directly. Text files (code, "
        "markdown, JSON, CSV, etc.) are inlined as text. Use this when you "
        "need to actually inspect contents — for listing files, use bash "
        "`ls drive/` instead. Hard cap: 5MB per file, 10 files per call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Drive paths relative to the workspace drive root "
                    "(e.g. ['uploads/photo.jpg', 'data/report.pdf']). "
                    "A leading 'drive/' is accepted and stripped."
                ),
                "minItems": 1,
                "maxItems": 10,
            },
        },
        "required": ["paths"],
    },
}

FILE_WRITE_TOOL_SPEC = {
    "name": "file_write",
    "description": (
        "Create a new text file in the workspace drive, or completely overwrite "
        "an existing one, with the exact `content` you provide. Parent "
        "directories are created automatically. Use this to author a file from "
        "scratch (source code, an HTML scaffold copy, a spec/plan note, "
        "JSON/config) or to replace one wholesale. For a SURGICAL change to part "
        "of an existing file, prefer `file_edit` — it patches just the target "
        "region "
        "instead of you re-sending the whole file, so it's faster and can't "
        "clobber unrelated lines. Text/UTF-8 only; binary or media files come "
        "from generate_*/download_url, not this. The write is persisted to the "
        "drive, so file_read, other tools, and later steps see it immediately. "
        "Returns the drive path, byte size, exact character count (`chars` — "
        "use it to check a budget-capped media prompt without re-emitting the "
        "text; see `prompt_path` on generate_image/video), and line count."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Drive path to write (e.g. '_jobs/<job_id>/game.js'). A "
                    "leading 'drive/' is accepted and stripped; '..' segments "
                    "are rejected."
                ),
            },
            "content": {
                "type": "string",
                "description": "Full file content to write, verbatim.",
            },
        },
        "required": ["path", "content"],
    },
}

FILE_EDIT_TOOL_SPEC = {
    "name": "file_edit",
    "description": (
        "Make a targeted, atomic edit to an existing text file in the workspace "
        "drive — the agentic way to evolve code in place without rewriting the "
        "whole file. Two modes:\n"
        "• STRING mode (preferred, robust): give `old_string` + `new_string`. "
        "`old_string` must match the file EXACTLY — including whitespace and "
        "indentation — and must be UNIQUE in the file, or the edit is rejected "
        "so you never patch the wrong place. Add surrounding context to make a "
        "short anchor unique, or pass `replace_all: true` to change every "
        "occurrence (e.g. renaming a symbol).\n"
        "• LINE mode: give `start_line` + `end_line` (1-based, inclusive) and "
        "`new_string` to replace exactly that line range; an empty `new_string` "
        "deletes the range. Use when a clean unique anchor is awkward.\n"
        "Returns a unified diff (with line numbers) of what changed, plus the "
        "new size and line count. The change is persisted to the drive. To "
        "create a file or replace it entirely, use `file_write` instead. Tip: read "
        "the file first (file_read, or `bash 'grep -n …'` to find line numbers) "
        "so your anchor is exact and unique."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Drive path to edit (e.g. '_jobs/<job_id>/game.js'). A "
                    "leading 'drive/' is accepted and stripped."
                ),
            },
            "old_string": {
                "type": "string",
                "description": (
                    "STRING mode: exact text to find and replace. Must be "
                    "unique in the file unless `replace_all` is true. Omit when "
                    "using LINE mode."
                ),
            },
            "new_string": {
                "type": "string",
                "description": (
                    "Replacement text. Required. In LINE mode this is the text "
                    "that replaces the line range (empty string deletes it)."
                ),
            },
            "replace_all": {
                "type": "boolean",
                "description": (
                    "STRING mode: replace EVERY occurrence of `old_string` "
                    "instead of requiring a unique match (default false)."
                ),
            },
            "start_line": {
                "type": "integer",
                "description": (
                    "LINE mode: 1-based first line to replace (use with "
                    "`end_line`, instead of `old_string`)."
                ),
            },
            "end_line": {
                "type": "integer",
                "description": "LINE mode: 1-based last line to replace, inclusive.",
            },
        },
        "required": ["path", "new_string"],
    },
}

DOWNLOAD_URL_TOOL_SPEC = {
    "name": "download_url",
    "description": (
        "Download a file from a URL via plain HTTP GET and save it to the "
        "workspace drive. Use this for images, PDFs, CSVs, etc. Returns the "
        "saved `drive_path` (sign it with `drive_url` only if you need a URL). "
        "Does NOT resolve share links (Google Drive/Dropbox/YouTube) — only "
        "direct HTTP(S) URLs work. Hard cap: 50MB per file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Direct file URL (http:// or https://)",
            },
            "path": {
                "type": "string",
                "description": (
                    "Where to save in the drive. Either a full path with "
                    "filename ('data/report.pdf') or a directory ending in '/' "
                    "('downloads/') — in the latter case the filename is "
                    "inferred from the URL."
                ),
            },
        },
        "required": ["url", "path"],
    },
}

DRIVE_URL_TOOL_SPEC = {
    "name": "drive_url",
    "description": (
        "Mint a fresh, time-limited public URL for a file already in the "
        "workspace drive. Use this to hand a drive file to a tool that needs "
        "a real URL — most importantly the `media` tool, whose image/video "
        "inputs (e.g. 'image_url', 'image_urls') must be URLs, not drive "
        "paths. Given a drive-relative path like 'uploads/logo.png', returns a "
        "signed https:// URL anyone can fetch until it expires. Free. (To go "
        "the other way — save a URL into the drive — use `download_url`.)"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Drive path relative to the workspace drive root "
                    "(e.g. 'uploads/logo.png'). A leading 'drive/' is accepted "
                    "and stripped."
                ),
            },
            "ttl": {
                "type": "integer",
                "description": "Seconds until the URL expires (60-86400, default 3600).",
            },
        },
        "required": ["path"],
    },
}


DRIVE_PULL_TOOL_SPEC = {
    "name": "drive_pull",
    "description": (
        "Download a file from the workspace drive onto local disk so you can "
        "read it with `bash` (cat, ffmpeg, a script). `file_read`, `generate_*`, "
        "and subagent inputs fetch drive files automatically — you only need "
        "`drive_pull` to reach a drive file from RAW bash that you did NOT create "
        "this run (e.g. an asset an earlier job saved). After it returns ok, the "
        "file is at `drive/<path>`. Files you produce this run (generate_*, "
        "download_url, tool outputs) are already local — no pull needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Drive path relative to the workspace drive root "
                    "(e.g. 'research/notes.md'). A leading 'drive/' is accepted "
                    "and stripped."
                ),
            },
        },
        "required": ["path"],
    },
}


DESCRIBE_SUBAGENT_TOOL_SPEC = {
    "name": "describe_subagent",
    "description": (
        "Look up the EXACT inputs a subagent `target` expects — its declared "
        "input schema (field names, types, which are required) and its output "
        "shape. ALWAYS call this before `run_subagent` for any skill `target`: "
        "run_subagent does NOT show you the target's schema, so without this "
        "you'd be guessing field names and a wrong shape fails validation and "
        "wastes a turn. `target` takes the same refs as run_subagent — a "
        "`references/*.md` bundle path, `skill`, `skillpack/skill`, or "
        "`workspace/skillpack/skill`. A `.md` / inline-prompt target has no "
        "declared schema (free-form inputs) and this will tell you so."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "The subagent to describe: a `references/*.md` bundle path "
                    "or a skill ref (`skill`, `skillpack/skill`, "
                    "`workspace/skillpack/skill`)."
                ),
            },
        },
        "required": ["target"],
    },
}


RUN_SUBAGENT_TOOL_SPEC = {
    "name": "run_subagent",
    "description": (
        "Run an isolated subagent and get its result back. A subagent run is "
        "its own job (own context window, own cost, linked to this run as a "
        "child), so use it to hand a self-contained stage of work to a fresh "
        "agent instead of doing everything in this context.\n\n"
        "BEFORE calling this with a skill `target`, call `describe_subagent` "
        "first to get the target's exact input schema, then pass `inputs` in "
        "that shape — this tool does not surface the target's schema, so "
        "guessing field names fails validation.\n\n"
        "Give EITHER `target` or `prompt`:\n\n"
        "`target` runs an existing skill or bundle prompt file:\n"
        "  - `references/foo.md` (any bundle-relative `*.md` path) — runs that "
        "markdown file as the system prompt of a fresh subagent, in THIS "
        "skillpack. The path is relative to your own skill directory. Use this "
        "for pipeline stages whose prompt lives in your `references/`.\n"
        "  - `skill` — another skill in this skillpack.\n"
        "  - `skillpack/skill` — a skill in another skillpack in this workspace.\n"
        "  - `workspace/skillpack/skill` — a public skill in another workspace.\n\n"
        "`prompt` runs a one-off inline system prompt as a subagent in this "
        "bundle's context, with the built-in tools and a free-form "
        "`set_output`. Use this for a quick self-contained stage you don't want "
        "to spell out as a file or skill.\n\n"
        "`inputs` is passed to the child verbatim (it reads them as its own "
        "inputs). For a `.md` or inline-prompt subagent there's no declared "
        "schema, so pass whatever that prompt expects — any file-shaped values "
        "(URLs, drive paths) are staged and attached for the child to look at."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "What to run: a `references/*.md` bundle path (isolated "
                    "subagent in this skillpack), or a skill ref "
                    "(`skill`, `skillpack/skill`, `workspace/skillpack/skill`). "
                    "Mutually exclusive with `prompt`."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "An inline system prompt to run as a one-off subagent in "
                    "this bundle's context. Mutually exclusive with `target`."
                ),
            },
            "inputs": {
                "type": "object",
                "description": (
                    "Inputs passed to the child verbatim, as a JSON object (NOT "
                    "a JSON string). For a skill `target`, match the shape from "
                    "`describe_subagent`. For a `.md` / inline-prompt target "
                    "there's no declared schema — pass whatever that prompt "
                    "expects; file-shaped values (URLs, drive paths) are staged "
                    "and attached for the child to look at."
                ),
                "additionalProperties": True,
            },
            "version": {
                "type": "integer",
                "description": (
                    "Pin a skill `target` to a specific deployment version of "
                    "its skillpack (e.g. 3). Omit to use the active deployment. "
                    "Only valid with a skill-ref `target` — not with a `.md` "
                    "path or `prompt`, which run against your own deployment."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Seconds to wait for the child (default and ceiling 1800; "
                    "higher values are clamped)."
                ),
            },
        },
    },
}


_MEMORY_NOTE = (
    "Workspace MEMORY is a shared brain across every skill in this workspace: a "
    "product/subject researched once is reused by later jobs (any skill) instead "
    "of re-researched. Store only STABLE facts that outlive this job — an entity's "
    "appearance, brand colors/logo, audience, tone; a durable user preference; or "
    "a recorded decision. NEVER store per-job creative choices (concept, shot "
    "plan, layout, headline, script, prompt text, chosen aspect ratios). "
    "Identity hints (entity_key + content_hash) for this job are provided in the "
    "first message — key your writes with them so the next run matches."
)

MEMORY_SEARCH_TOOL_SPEC = {
    "name": "memory_search",
    "description": (
        "Look up existing workspace memory BEFORE doing expensive research. "
        "Returns matching records (best first — relevance is fused from exact "
        "identity, text and semantic similarity, then shaped by recency and "
        "importance) with their `record` payload and a `fresh` flag — a fresh "
        "hit means you can REUSE the record and skip re-researching. Match by "
        "`kind` + `key` (an entity_key/alias) and/or `content_hash`, and/or a "
        "free-text `query` over titles, summaries and tags. " + _MEMORY_NOTE
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "description": (
                    "Record kind to search, e.g. 'product_brief', 'game_brief', "
                    "'brand_kit', 'store_listing', 'research', 'user_preference'."
                ),
            },
            "key": {
                "type": "string",
                "description": (
                    "An entity_key or alias to match exactly — e.g. the "
                    "`entity_key` identity hint from the first message, a "
                    "normalized URL, or a product slug."
                ),
            },
            "content_hash": {
                "type": "string",
                "description": (
                    "The job's inputs fingerprint (the `content_hash` identity "
                    "hint). When given, a hit is only `fresh` if the stored hash "
                    "matches — i.e. the inputs haven't changed."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Free-text search over titles, summaries, kinds and tags "
                    "(plus semantic similarity when available). Use a few "
                    "descriptive words, e.g. 'coffee mug brand colors'."
                ),
            },
            "mtype": {
                "type": "string",
                "enum": ["semantic", "episodic", "procedural"],
                "description": "Restrict to a memory type. Omit to search all.",
            },
            "scope": {
                "type": "string",
                "enum": ["entity", "skillpack", "workspace"],
                "description": "Restrict to a sharing scope. Omit to search all.",
            },
            "limit": {"type": "integer", "description": "Max results (default 8)."},
        },
        "required": [],
    },
}

MEMORY_GET_TOOL_SPEC = {
    "name": "memory_get",
    "description": (
        "Fetch one memory record in full by its `id` (from a memory_search hit or "
        "an injected memory_id). Use when a search preview was truncated and you "
        "need the complete `record`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The memory record id (uuid)."},
        },
        "required": ["id"],
    },
}

MEMORY_PUT_TOOL_SPEC = {
    "name": "memory_put",
    "description": (
        "Save (or update) a record in workspace memory so future jobs reuse it. "
        "A keyed record (semantic/procedural with an `entity_key`) is UPSERTED — "
        "calling it again for the same `kind`+`entity_key` overwrites and bumps "
        "the version, so just write the current best version wholesale. Episodic "
        "records (a logged event) append. " + _MEMORY_NOTE
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "description": (
                    "What this record is: 'product_brief', 'game_brief', "
                    "'brand_kit', 'store_listing', 'research', 'user_preference', "
                    "'approval', … Choose a stable, reusable name."
                ),
            },
            "record": {
                "type": "object",
                "description": (
                    "The STABLE payload to store (a JSON object). For a brief: the "
                    "reusable facts only (appearance, colors, logo, audience, "
                    "tone) — no creative-meta."
                ),
                "additionalProperties": True,
            },
            "entity_key": {
                "type": "string",
                "description": (
                    "Canonical identity for this subject — use the `entity_key` "
                    "identity hint from the first message (a hero-image hash or "
                    "normalized URL), or author a stable slug. Required to make a "
                    "record reusable by exact lookup. Omit only for episodic logs."
                ),
            },
            "mtype": {
                "type": "string",
                "enum": ["semantic", "episodic", "procedural"],
                "description": (
                    "semantic = entity record/brief (default); procedural = "
                    "durable preference/rule (often pinned); episodic = a logged "
                    "event (appends, no upsert)."
                ),
            },
            "scope": {
                "type": "string",
                "enum": ["entity", "skillpack", "workspace"],
                "description": (
                    "How widely it applies: entity (about one subject, default), "
                    "workspace (brand kit / preferences shared by all skills)."
                ),
            },
            "title": {"type": "string", "description": "Human label for the memory browser (e.g. the product/game name)."},
            "summary": {
                "type": "string",
                "description": (
                    "STRONGLY RECOMMENDED: a 1–3 sentence natural-language gist "
                    "of the record ('Ceramic coffee mug by Acme; matte black, "
                    "gold logo; audience: office workers'). This is what future "
                    "text/semantic search matches against — a record without it "
                    "is only findable by exact key."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A few search labels, e.g. ['mug', 'acme', 'drinkware'].",
            },
            "importance": {
                "type": "number",
                "description": (
                    "0..1 salience for ranking (default 0.5). Reserve >0.8 for "
                    "facts that must surface (allergies-grade), <0.3 for minor "
                    "notes."
                ),
            },
            "supersedes": {
                "type": "string",
                "description": (
                    "id (uuid) of an existing record this one REPLACES (it gets "
                    "soft-deleted and linked to the new record). Use when "
                    "correcting a record stored under a different kind/key — "
                    "same-key re-puts already overwrite, no need for this."
                ),
            },
            "content_hash": {
                "type": "string",
                "description": (
                    "The inputs fingerprint (the `content_hash` identity hint) so "
                    "a later run can tell if the inputs changed (staleness)."
                ),
            },
            "source_url": {"type": "string", "description": "Source URL the record was derived from, if any."},
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra keys this subject can also be found by (e.g. a normalized product name).",
            },
            "pinned": {
                "type": "boolean",
                "description": (
                    "If true, this record is injected into EVERY future job in "
                    "scope (use for workspace preferences / brand kit). Default false."
                ),
            },
            "stale_at": {
                "type": "string",
                "description": (
                    "Optional ISO-8601 timestamp after which the record is "
                    "considered stale and won't be reused (use for perishable "
                    "facts like a current offer)."
                ),
            },
        },
        "required": ["kind", "record"],
    },
}

MEMORY_FORGET_TOOL_SPEC = {
    "name": "memory_forget",
    "description": (
        "Soft-delete one memory record that is wrong, obsolete or harmful — it "
        "stops matching searches and injections immediately (an audit copy is "
        "kept briefly, then purged). Use when you confirmed a stored record is "
        "incorrect for its key and you are NOT writing a replacement (when you "
        "are, prefer memory_put with `supersedes`, or a same-key re-put)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The memory record id (uuid) to forget."},
            "reason": {
                "type": "string",
                "description": "Short reason ('outdated listing', 'wrong product'). Kept for the audit trail.",
            },
        },
        "required": ["id"],
    },
}


def set_output_tool_spec(output_schema: dict) -> dict:
    from .schema_dialect import to_output_jsonschema

    return {
        "name": "set_output",
        "description": (
            "Record the final structured output for this job. Calling this "
            "ends the run. The argument must match the schema below."
        ),
        # Output schemas omit `required`: every declared property is mandatory.
        "input_schema": to_output_jsonschema(output_schema),
    }


def _collect_builtin_specs() -> list[dict]:
    """Auto-discover every `*_TOOL_SPEC` dict defined in this module.

    Define a new top-level `FOO_TOOL_SPEC = {...}` above and it shows up
    automatically in the agent's tool list and in the generated docs.
    Order follows module definition order (Python ≥3.7 dict insertion).
    Dispatch for the new name still needs a handler in
    `agent_runner.run_agent`.
    """
    import sys as _sys
    out: list[dict] = []
    for name, val in vars(_sys.modules[__name__]).items():
        if not name.endswith("_TOOL_SPEC"):
            continue
        if not isinstance(val, dict):
            continue
        if "name" not in val or "input_schema" not in val:
            continue
        out.append(val)
    return out


BUILTIN_AGENT_TOOLS: list[dict] = _collect_builtin_specs()
