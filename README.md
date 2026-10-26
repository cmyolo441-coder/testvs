<div align="center">

<img src="assets/omh-logo.png" alt="OMH logo" width="120">

# OMH

**A fast, dependency-free terminal AI agent — rebuilt from oh-my-hermes.**

`omh` streams, calls tools in parallel, balances a pool of models, and starts
in the time it takes Python to import one small package.

[Quick start](#quick-start) · [Load balancing](#load-balancing) · [Tools](#tools) ·
[Commands](#commands) · [Configuration](#configuration) · [Tests](#tests)

</div>

---

## Why this exists

The upstream `oh-my-hermes` repository grew into a 149 MB install with 1,182
generated Python modules, translated READMEs, benchmark fixtures and a
skill catalogue you could not read. The useful ideas — a model per lane, skills
on demand, evidence before “done” — were buried under that.

OMH keeps those three ideas in a small runnable harness, and keeps the upstream
packages around for reference:

| | upstream | omh |
| --- | --- | --- |
| wheel size | 149 MB install | 62 KB |
| Python modules | 1,182 | 23 runnable (+ `src/` reference copy) |
| source lines | ~500k | ~5,300 runnable |
| runtime dependencies | none (but wraps Hermes) | **none** |
| start-up | install + setup + gateway | `python main.py` |
| provider support | via Hermes | OpenAI-compatible, Anthropic, Ollama, mock |

Everything is stdlib: `urllib` for streaming, `threading` for hedging and
parallel tools, `readline` for history, `unittest` for tests.

## Quick start

```sh
git clone https://github.com/rlaope/oh-my-hermes.git
cd oh-my-hermes

# zero-key demo: every part of the harness runs on the mock provider
python main.py --offline

# or with a real key
export OPENAI_API_KEY=sk-...
python main.py

# keys, models and base URLs can also live in config.toml:
#   cp config.toml ~/.omh/config.toml   (or: python main.py --init)
python main.py --init
```

Install the `omh` command (optional, still no dependencies):

```sh
python -m pip install -e .
omh
```

One-shot mode:

```sh
python main.py -p "summarise this repo and list the riskiest files" --yes
echo "run the tests and fix what fails" | python main.py --yes
```

```
  ◆ omh  terminal agent harness   v1.0.0
  model    auto · balanced · 3 lanes
  pool     7 models · 2 providers · balanced
  cwd      /home/you/project

  /help commands   ! shell   esc / ctrl-c cancel

❯ refactor the router and prove it still balances
  ◆ omh  gpt-4o-mini
▌ Plan
- read `omh/router.py` and its tests
- extract scoring, keep behaviour
...
  ◆ read_file  omh/router.py
  ✔ read_file  ok
...
  ◆ gpt-4o-mini · 4.2s · 8,120 tok · $0.0021 · 3 steps
```

## Features

- **Streaming everywhere.** Text is rendered line by line as it arrives; the
  first token paints immediately, with a spinner on stderr that never dirties
  the transcript.
- **Parallel tools.** Independent calls in one assistant turn run on a thread
  pool; approval prompts stay on the main thread so they never interleave.
- **A real load balancer.** Weighted, latency-aware routing with failover,
  cooldowns and optional hedged requests (see below).
- **Two provider dialects.** OpenAI-compatible (`/chat/completions`) and
  Anthropic (`/v1/messages`), both streaming with tool calls.
- **Approval modes.** Writes and shell commands ask by default; `--yes` or
  `/approval auto` lets the agent run unattended.
- **Sessions.** Every turn is saved as JSON under `~/.omh/sessions`; resume with
  `--resume`, `/resume`, or `--session <id>`.
- **Skills on demand.** Four readable markdown prompt packs in `skills/`, loaded
  with `/skill review`.
- **Tiny, inspectable surface.** 23 modules, ~5,300 lines, no magic, no plugin
  loader, no generated catalogues.
- **Works offline.** `omh --offline` exercises streaming, tool calls and the
  renderer with the built-in mock provider.

## Load balancing

The pool is every model whose provider has credentials. Three strategies:

| strategy | pick order |
| --- | --- |
| `balanced` (default) | `weight / (time-to-first-token + 0.15)`; unmeasured lanes get a 4× exploration bonus |
| `round_robin` | smooth weighted round robin (nginx algorithm) |
| `priority` | configuration order: cheap and predictable |

On top of that, every turn gets:

- **Failover.** A provider error, 401/403, 429, timeout or HTTP 5xx puts that
  lane in an exponential cooldown (300 s for auth failures) and the next
  candidate is tried *before any text is shown*.
- **Hedging.** Set `router.hedge_delay_s` (e.g. `0.8`): if the chosen lane has
  not produced a token within that time, a second lane starts in parallel and
  the first token wins. The loser is cancelled, so it costs nothing but a
  partial request.
- **Stats.** `/balance` shows per-lane requests, failures, EWMA TTFT, tokens and
  cooldown state; `/strategy` switches modes live.

```json
{
  "router": {
    "strategy": "balanced",
    "max_attempts": 3,
    "cooldown_s": 20,
    "hedge_delay_s": 0.8,
    "max_parallel_tools": 4
  }
}
```

Pin one lane with `/model gpt-4o-mini`, one provider with `/provider anthropic`,
or leave the pool on `auto` to spread traffic across everything that is ready.

## Tools

| tool | approval | what it does |
| --- | --- | --- |
| `read_file` | no | numbered lines, `start_line`/`end_line`, 400 KB cap |
| `write_file` | **yes** | create or overwrite, creates parent dirs |
| `edit_file` | **yes** | exact-string replace, uniqueness check, returns a diff |
| `list_dir` | no | folders first, file sizes, hidden optional |
| `glob` | no | `**/*.py` style search, skips `.git`/`node_modules`/venvs |
| `grep` | no | regex content search, `path:line: text`, 500 result cap |
| `run_shell` | **yes** | captured stdout/stderr + exit code, hard timeout |
| `web_fetch` | no | http(s) page → readable text, 2 MB / 100 k char caps |

All file paths are resolved inside the workspace; `..` escapes are rejected.
Tool output is truncated (`max_tool_output`, default 20 000 chars) and fed back
to the model as a `tool` message. A failing tool is a normal observation, not a
crash.

## Safety

- `write_file`, `edit_file` and `run_shell` ask for approval by default. `--yes`
  or `/approval auto` runs unattended; `/approval deny` makes the agent
  read-only.
- File tools are sandboxed to the workspace: absolute paths and `..` escapes are
  rejected. `run_shell` is inherently not sandboxed — approval is the boundary,
  so only use auto mode in a repository you trust.
- Fetched pages and tool output are untrusted input. The system prompt tells the
  model to treat them as data, never as instructions.
- Sessions are plain JSON in `~/.omh/sessions`. API keys are never written
  there; config only stores the name of the environment variable.

## Commands

Inside the REPL:

```
/help                 command list and key bindings
/model [name|auto]    pin a model, or load balance the pool
/models               weights, context, price, credential status
/provider [name]      restrict to one provider, or auto
/strategy <name>      balanced | round_robin | priority
/balance              live per-lane stats, cooldowns, EWMA TTFT
/tools                tool list and approval requirements
/skill [name|off]     load a markdown skill into the system prompt
/theme [name]         hermes | amber | mono
/approval [mode]      ask | auto | deny
/new                  start a fresh session
/sessions             list saved sessions
/resume [id|last]     load a session
/clear                forget this session's messages
/cost                 tokens, spend, model
/doctor               config, providers, credential health
/init                 write ~/.omh/config.toml
/save                 save now
/version              print version
/exit                 leave
```

`!<command>` runs a shell command directly in the workspace; `ctrl-c` cancels a
turn; `ctrl-d` exits.

## Configuration

Everything about models — ids, providers, base URLs, weights, prices and keys —
lives in [`config.toml`](config.toml). There is no model list in Python code.
OMH reads, in order:

1. the built-in `config.toml`
2. `~/.omh/config.toml`
3. `./.omh.toml` (project config; `OMH_CONFIG=/path` also works)
4. `OMH_*` environment variables
5. CLI flags

`.env` files in the workspace and `~/.omh/.env` are loaded without overwriting
real environment variables. `omh --init` copies the commented template to
`~/.omh/config.toml`; `omh --doctor` shows what is active.

```toml
model = ""                # ""/auto = load balance, or a model id/alias
approval_mode = "ask"
theme = "hermes"

[router]
strategy = "balanced"
max_attempts = 3
cooldown_s = 20.0
hedge_delay_s = 0.0
max_parallel_tools = 4

[providers.openai]
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
api_key = ""              # or "${MY_KEY}" or a literal key (do not commit it)

[providers.local]
kind = "openai"
base_url = "http://localhost:8000/v1"
requires_key = false

[[models]]
id = "gpt-4o-mini"
provider = "openai"
label = "gpt-4o-mini"
weight = 2.0
context = 128000
input_cost = 0.15
output_cost = 0.60

[[models]]
id = "qwen2.5-coder-32b"
provider = "local"
label = "qwen-coder"
weight = 3.0
context = 32768
```

Add a provider with one `[providers.<name>]` table and a model with one
`[[models]]` block, then restart (or `/model auto`). Keys can be a literal
`api_key`, an `api_key_env` variable name, or `api_key = "${VAR}"` which is
expanded from the environment.


Key environment variables:

| variable | meaning |
| --- | --- |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY` | provider credentials |
| `OMH_BASE_URL`, `OMH_API_KEY` | any OpenAI-compatible endpoint (vLLM, LM Studio, Together…) |
| `OMH_MODEL`, `OMH_PROVIDER`, `OMH_STRATEGY` | overrides |
| `OMH_APPROVAL`, `OMH_THEME`, `OMH_MAX_STEPS` | overrides |
| `OMH_HOME` | config/session root (default `~/.omh`) |
| `OMH_DEBUG=1` | re-raise internal errors instead of rendering them |

## Architecture

```
main.py                 entry point (python main.py)
config.toml             all models, providers, base URLs, keys, routing
omh/
  cli.py                argparse, REPL, slash commands, tables
  agent.py              stream → tool calls → repeat, parallel tools
  router.py             weighted balancing, failover, cooldown, hedging
  config.py             TOML config, env, .env, model pool
  default_config.toml   packaged copy of config.toml (wheel installs)
  providers/
    base.py             messages, tool calls, chunks
    http.py             urllib SSE client + error mapping
    openai.py           OpenAI-compatible streaming
    anthropic.py        Messages API streaming
    mock.py             offline demo provider
  tools/                read/write/edit, glob/grep, shell, web
  session.py            JSON sessions, resume, usage
  skills.py             markdown prompt packs
  theme.py              palette, markdown renderer, tables, banner
  ui.py                 spinner, tool cards, prompt, approvals
  errors.py             one error hierarchy
skills/                 review, debug, refactor, test-first
src/                    upstream oh-my-hermes packages, kept for reference
tests/                  stdlib unittest suite
```

`src/` is an unmodified copy of upstream `src/` (1,255 files, 23 packages) so
the routing, coding, evidence and workflow code is available to read or port.
It is **not** imported by `main.py` and is not included in the wheel; see
[`src/UPSTREAM.md`](src/UPSTREAM.md).

The loop is deliberately boring: build the message list, trim it, stream a
completion, append the assistant message, run any tool calls in parallel,
append their results, repeat until the model stops or `max_steps` is reached.
History is trimmed at user-turn boundaries so tool-call/result pairs are never
split.

## Tests

No pytest required:

```sh
python -m unittest discover -s tests -v
```

The suite covers SSE parsing and tool-call accumulation, both provider
dialects, all three routing strategies, cooldowns and hedging, every tool
including workspace escapes, the agent loop with parallel tool execution,
session persistence, the markdown renderer, config precedence, and CLI
smoke tests (including a real `python main.py --offline` run).

## Provenance

OMH is a ground-up rewrite that reuses ideas (and the turquoise skin palette)
from [`rlaope/oh-my-hermes`](https://github.com/rlaope/oh-my-hermes). The
installer scripts, docs site, benchmarks, translated READMEs and generated
catalogues are not part of the runnable CLI. The upstream `src/` packages are
kept verbatim under [`src/`](src/UPSTREAM.md) for reference, so the routing,
coding and workflow machinery is still in the repository; only the 23-module
`omh/` package and `config.toml` drive the agent.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 oh-my-hermes contributors.



