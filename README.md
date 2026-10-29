<div align="center">

```
███████╗██╗   ██╗██╗    ██╗    █████╗  ██████╗ ███████╗███╗ ██╗████████╗
██╔════╝██║   ██║██║    ██║   ██╔══██╗██╔════╝ ██╔════╝████╗██║╚══██╔══╝
█████╗  ██║   ██║██║    ██║   ███████║██║  ███╗█████╗  ██╔██╗██║  ██║
██╔══╝  ██║   ██║██║    ██║   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║  ██║
██║     ╚██████╔╝██████╗█████╗██║  ██║╚██████╔╝███████╗██║ ╚████║  ██║
╚═╝      ╚═════╝ ╚═════╝╚════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝  ╚═╝
```

<h1>FullAgent <sup><code>v4.0.0</code></sup></h1>

**Advanced Terminal AI Agent — Pure Python, Real Code, World-Class TUI**

<p>
  <a href="https://github.com/cmyolo441-coder/py.test/releases"><img src="https://img.shields.io/github/v/release/cmyolo441-coder/py.test?label=release&color=bd93f9&style=flat-square" alt="release"></a>
  <a href="https://github.com/cmyolo441-coder/py.test"><img src="https://img.shields.io/badge/python-3.9%2B-8be9fd?style=flat-square&logo=python&logoColor=white" alt="python"></a>
  <a href="https://github.com/cmyolo441-coder/py.test"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows%20%7C%20Termux-50fa7b?style=flat-square" alt="platform"></a>
  <a href="https://github.com/cmyolo441-coder/py.test/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/pure--python-%23FFB86C?style=flat-square" alt="pure python"></a>
  <a href="https://github.com/cmyolo441-coder/py.test"><img src="https://img.shields.io/github/stars/cmyolo441-coder/py.test?style=flat-square&color=f1fa8c" alt="stars"></a>
  <a href="https://github.com/cmyolo441-coder/py.test/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-6272a4?style=flat-square" alt="license"></a>
</p>

*`Event-Sourced Kernel` · `Goal Contracts` · `Self-Healing` · `Temporal Kernel` · `40+ Slash Commands` · `16 Tools` · `Kios API`*

<p>
  <a href="#-install-binary--curl-one-liner"><b>Install</b></a> •
  <a href="#-run"><b>Run</b></a> •
  <a href="#-what-it-can-do"><b>Features</b></a> •
  <a href="#-slash-commands"><b>Commands</b></a> •
  <a href="#-layout"><b>Layout</b></a>
</p>

</div>

---

> **Qoder-style chrome ✨ — naya welcome card, live agent board aur status HUD!**  
> `python main.py` chalate hi block logo + "Tips for getting started" card dikhta hai,
> aur kaam ke dauraan neeche ek permanent HUD rehta hai: mode chip, skills counter,
> live parallel-subagent board, model/effort/context footer aur agent count.

```
 ███████╗ █████╗                    ╭───── Tips for getting started ─────╮
 ██╔════╝██╔══██╗                   │ 1. Use AGENTS.md files to customize │
 █████╗  ███████║                   │ 2. /help for more information       │
 ██╔══╝  ██╔══██║                   │ 3. Ask coding questions, edit code… │
 ██║     ██║  ██║                   │ 4. Be specific for the best results │
 ╚═╝     ╚═╝  ╚═╝                   ╰─────────────────────────────────────╯
  FullAgent CLI v4.0.0
```

While a swarm is running the frame shows the Qoder-style board and footer:

```
 ≡ 8 Agents (8 running) ───────────────────────────── ↓ to manage
 ◉ Explore · Read fullagent/autopilot.py
 ◉ Code    · Edit fullagent/tui.py
   … +6 more · ↓ to manage
 COLLAB  ⇧Tab pilot · ◎ 8 running · 0 queued ── 0 skills
╭─ FullAgent ── ● DeepSeek V4.1 Flash Free ── ◈ extra high ── ◉ ctx 3% ─╮
│ ❯ type anything… the agent reads, writes, edits files, runs commands │
╰ ⏎ send · ⇧⏎ newline · / cmds · /diff live · ^T models · ^E effort ──╯
 ● DeepSeek V4.1 Flash Free · extra high · 1m context · ctx ░░░░░░░░░░ 3% · ~/Desktop
                           ◎ feat/tui-chrome · ◎ 8 agents | 8 running
```

| Key | Action |
|---|---|
| `Shift+Tab` | Toggle Pilot (L4 — safe tools run without asking; deletes still ask) |
| `Ctrl+O` | Expand / collapse tool output (`… +N more` rows) |
| `↓` on an empty prompt | Open the live agent dashboard (`/agents`) |
| `Ctrl+T` / `Ctrl+E` | Model / effort picker |

Every value in that frame is live state, not decoration. The board and both
agent counters read `ParallelSubagents.snapshot()`: only workers that are
running or queued right now are counted, `8 running` appears only when eight
workers are genuinely executing (the pool is 8-wide, matching the largest
`TaskGroup` batch), queued workers are shown separately, and the badge
disappears the moment the swarm finishes. Retained history from earlier
batches is never counted as live. The footer branch is the real
`git rev-parse --abbrev-ref HEAD` of the cwd, the mode chip names the real
autonomy level from `AUTONOMY_LEVELS` (L0 Observe … L5 Autonomous), the
skills counter is the real `SKILL.md` + forge registry (0 until you install
or author one), and the context meter is the live token estimate against
the model's real window (1m here). `<repo>/AGENTS.md` (plus ancestors and
`~/.fullagent/AGENTS.md`) is genuinely loaded into the system prompt as the
`PROJECT INSTRUCTIONS` context section — edit it and the next model call
sees the change.

The prompt is a **double-line box** (not full-screen). One application runs
for the whole session, so the box **never disappears** — while the model
works, the bottom border shows a live animated status:

```
╰ ⠹ thinking…  ·  Ctrl+C cancel ────────────────────────────────────────────╯
```

Tokens stream live above the box, tool calls appear as `⚙ name args` with
`✓`/`✗` results, and each turn ends with a stats line (`2.3s · 512→87 tokens`).

Streaming updates the unfinished-line preview on every token; only terminal
redraws are throttled (about 33 fps). Complete lines print without waiting for
the next token. Rapid submissions cannot overlap an active turn, including
final rendering and session saving. Completed file-change output prints
without artificial animation delays.

<details>
<summary><b>🖼️ Terminal Banner Preview (after <code>python main.py</code>)</b></summary>

```
 ███████╗ █████╗                    ╭───── Tips for getting started ──────────╮
 ██╔════╝██╔══██╗                   │ 1. Use AGENTS.md files to customize beh… │
 █████╗  ███████║                   │ 2. /help for more information            │
 ██╔══╝  ██╔══██║                   │ 3. Ask coding questions, edit code, run… │
 ██║     ██║  ██║                   │ 4. Be specific for the best results      │
 ╚═╝     ╚═╝  ╚═╝                   ╰──────────────────────────────────────────╯
  FullAgent CLI v4.0.0
  API key · built-in (Kios API)
  DeepSeek V4.1 Flash Free  ·  high
  ~/Desktop  ·  session a1b2c3d4
 ❯ model  DeepSeek V4.1 Flash Free   effort  high   autonomy  L3   session  a1b2c3d4
   / commands · Ctrl+T models · Ctrl+E effort · Shift+Tab pilot · Ctrl+O expand output
```

*Wide terminals get the side-by-side card; under 78 columns the tips card
stacks under the logo so nothing wraps or clips.*

</details>

## Install (binary — curl one-liner)

Single-file binaries — no Python needed:

| Platform | Install |
|---|---|
| Linux x64 | `curl -fsSL https://raw.githubusercontent.com/cmyolo441-coder/py.test/main/install.sh \| bash` |
| macOS (Apple Silicon) | download `fullagent-darwin-arm64` from the [latest release](https://github.com/cmyolo441-coder/py.test/releases/latest), `chmod +x`, move to /usr/local/bin |
| Windows x64 | download `fullagent-windows-x64.exe`, rename to `fullagent.exe`, put on PATH |
| Termux (Android) | `curl -fsSL https://raw.githubusercontent.com/cmyolo441-coder/py.test/main/install-termux.sh \| bash` |
| Any OS with Python 3.9+ | `pip install git+https://github.com/cmyolo441-coder/py.test.git` |

`install.sh` grabs the latest release binary automatically and falls back to
a pip install from source when no binary is published yet. Set `INSTALL_DIR`
to install elsewhere (e.g. `INSTALL_DIR=$HOME/.local/bin curl -fsSL … | bash`).

## Run

```bash
python main.py
```

(ya `python -m fullagent` — dono same hain)

## Provider and rate limits

`DeepSeek V4.1 Flash Free` (Kios API) is the default configured model and works without
manually setting an API key. The app does not impose a five-requests-per-minute
client cap; when Kios returns a rate-limit response, it honors `Retry-After`
or waits with backoff and retries until the request succeeds or you cancel
with `Ctrl+C`. Kios still controls its own free-tier capacity and quotas, so
the app cannot guarantee literally unlimited service.

Browser automation is optional. Install it with `pip install 'fullagent[browser]'`
and `python -m playwright install chromium`; if system Chromium is already on
`PATH`, FullAgent can use it directly.

## What it can do

- **Files** — read (with line numbers), write, exact-string edit, list, info,
  create dirs, copy, move, delete
- **Shell** — run any bash command, capture exit code + stdout + stderr
- **Search** — regex search through file contents (ripgrep-style), glob files
- **Web** — fetch URLs, search the web
- **Agent loop** — the model keeps calling tools until the task is genuinely
  done (up to 40 iterations), then summarizes
- **Temporal Kernel** — every message, tool call, result, and cost is an
  immutable, content-addressed event in an append-only log. State is a pure
  fold of that log, so you can rewind, fork, replay, and verify the timeline
- **Goal contracts** — a goal is a structured object with done-criteria
  clauses and anti-clauses; distance-to-done is a computed number, not a vibe
- **Memory** — closed tasks compress into structured episode records; failed
  approaches land in a dead-end ledger and are blocked deterministically
- **Judge** — claims are verified against reality with deterministic
  predicates (exit codes, file checks, regex) — never the model's own word
- **Parallel subagents** — 14 focused specialists run concurrently (up to
  four live workers; groups can queue up to eight tasks) with per-task roles,
  independent context, bounded steps, live status, cancellation, named batches,
  consolidated result collection, and an audited handoff. `TaskGroup` handles
  parallel read-only review/research; `TaskWriteGroup` enables parallel coding
  through the existing permission gate, with distinct workspace path scopes,
  shell/deletion tools withheld, and a recovery snapshot before every delegated
  write. Workers stay on the selected configured model unless a model is
  explicitly selected; automatic model switching is disabled. The input box
  stays live while they run — follow-up messages are queued and drained one
  at a time after the current turn, never as concurrent `run_turn` loops.
  `TaskCollect`/`TaskOutput` waits on live workers are excluded from the
  turn's tool-time budget, so a parent turn cannot be stopped while its own
  subagents are still working.
- **Focus Mode** — deep work: `/focus 10` arms auto-continuation and the
  agent keeps working turn after turn until the goal closes, progress
  stalls, or the budget pauses. Every continuation decision is a sealed
  kernel event (`focus.tick` / `focus.stop`)
- **Rendered replies** — `/render on` streams the reply live in the
  border and prints the finished answer as rich Markdown
- **Instant triage** — tool errors show the healer's root-cause
  classification right on the result line
- **Audit export** — `/export md|html` writes a self-contained session
  report: timeline, tool stats, judge verdicts, per-model usage
- **Forecast** — `/forecast` projects turns-to-done from measured goal
  velocity and tokens-per-turn (numbers, not vibes)
- **Provider health** — `/health` shows model errors. Automatic model
  switching is disabled; provider failures stay on the selected model.
- **Notifications** — `/notify <webhook|file:path>` fires kernel events
  (goal closed, focus stop, workflow done…) to your sink
- **Session resume** — `/resume` lists branches/sessions; continue any
  of them with its conversation rebuilt from the event log
- **Turn scorecard** — every turn ends with deterministic quality
  metrics (errors, rework, verified claims) sealed as `turn.scorecard`
- **Live context meter** — the border shows real-time context-window
  usage (`ctx 37%`), and approvals show a real unified diff before you
  press `y`
- **AutoPilot** — the agent decides **for itself** what each turn needs and
  enables it automatically: goal mode when the request is a verifiable
  mission, real-time web when the question needs live data. Every decision
  is logged and shown live
- **Real-time web** — `web_search` hits DuckDuckGo with a Bing fallback and
  stamps every result with its retrieval time, so the agent answers with
  current facts, not stale knowledge

Risky tools (writes, edits, shell, delete, move, copy) ask for approval
**inside the app** — the bottom border becomes an approval bar:
press `y` (yes), `n` (no), or `a` (always). Toggle globally with `/approve`.
The **autonomy ladder** (`/autonomy 0-5`) controls how much the agent may do
without asking, from read-only observer to fully autonomous.

## Keys

| Key | Action |
|---|---|
| `Enter` | send |
| `Enter` while a turn is running | queue a follow-up; it starts after the current turn finishes |
| `Esc+Enter` | newline inside the box |
| `/` | slash-command completion menu |
| `Ctrl+T` or `/model` | model selector |
| `Ctrl+E` or `/effort` | effort selector |
| `↑↓` / `PgUp` / `PgDn` / `Tab` / `Home` / `End` | navigate selectors **and** the `/` completion menu |
| `Ctrl+R` | search input history |
| `Ctrl+L` | clear screen |
| `Ctrl+X Ctrl+E` | open input in $EDITOR |
| `Ctrl+C` | cancel a running turn (mid-stream) / clear input |
| `Ctrl+D` | quit |

## Slash commands

`/model` `/effort` `/help` `/history` `/new` `/save` `/approve`
`/reasoning` `/usage` `/clear` `/about` `/exit`

- `/agents` — rich crew dashboard; `/agents roles` shows specialist modes;
  `/agents detail <id>` inspects one worker; `/agents results [batch-id]`
  collects a batch; `/agents spawn [role] <read-only task>` starts one;
  `/agents wait [id|all]` waits up to 30 seconds; `/agents stop <id|all>`
  requests cancellation. Use natural-language tasks for parallel read-only work
  and approved, file-scoped coding delegation.

Event-log commands:

- `/goal set <statement> | <clause1> | <clause2>` — set a goal contract
  (prefix a clause with `!` to make it an anti-clause);
  `/goal done <clause>` · `/goal status` · `/goal clear`
- `/autonomy <0-5>` — observer → advisor → assistant → collaborator (default)
  → pilot → autonomous
- `/state` — live projection of the event log (cost, goal, dead-ends, verdicts)
- `/rewind <seq>` — rewind the timeline (bare `/rewind` lists recent seqs)
- `/fork [name]` — branch the timeline and continue on the fork
- `/verify` — verify the event log's Merkle spine
- `/memory` — recent episodes + dead-end ledger
- `/judge <type> <arg>` — deterministic check (`exit_code`, `file_exists`,
  `file_contains`, `file_matches`, `command_output_contains`), or pass a full
  JSON predicate
- `/auto [on|off|status]` — the AutoPilot self-routing brain (on by default)
- `/prompt [main|master|list]` — choose the system prompt: `main` (compact)
  or `master` (adds the optional `project.txt` specification when present)
- `/mastermind` — the prompt-coherence ledger (sealed prompts, gate,
  composed context, lineage)
- `/dashboard` — live observability: cost, goal, councils, router, spec,
  memory, health in one screen
- `/router` — task classification and routing history; model: DeepSeek V4.1 Flash Free
- `/spec` — speculative execution: prefetch stats + hit-rate
- `/recall <question>` — semantic (meaning-based) memory recall
- `/mission [start|tick|list|abandon]` — daemon mission control
- `/heal` — self-healing ledger: root causes captured + healed
- `/skills` — the skill forge: self-authored, safety-gated tools
- `/focus <1-20>` — deep-work mode: auto-continues until done · `/focus off`
- `/render [on|off]` — rendered-markdown replies (streaming stays live in the border)
- `/council <proposition>` — convene an adversarial debate
- `/analyze <path>` — static analysis: taint flows, complexity, cycles
- `/graph [index|query|impact]` — knowledge graph of code + session
- `/coverage` — real line-coverage ledger (sys.settrace)
- `/fuzz` — property-based fuzzing ledger: crashes + shrunk reproducers
- `/mutate <file> <suite-cmd>` — mutation testing: can your tests catch bugs?

## Effort levels

`low` · `medium` · `high` · `extrahigh` · `ultrahigh` — each raises max
tokens, temperature, and reasoning effort.

## Models & providers

**Kios API is the only configured provider.** Two models ship, all on Kios:

| Model | Tag | Context |
|---|---|---|
| `deepseek-v4.1-flash-free` (default) | free | 1.0m |
| `minimax-m3.1-flash-free` | free | 1.0m |

`deepseek-v4.1-flash-free` is the fastest Kios free route we measured
(~1s to first token); it is the default and runs with reasoning disabled so
answers stream immediately. MiniMax keeps its provider-default reasoning.
Unknown or retired model ids now fail immediately instead of retrying the
provider's 503 for ~14s.

Zero-config: a Kios API key is embedded in the build, so it works right after
installation with no setup. Override anytime via the `KIOS_API_KEY` environment
variable or `~/.fullagent/kios_api_key` (keep the file private); the env var
takes precedence, then the key file, then the embedded key.

| Provider | Base URL | Key required? |
|---|---|---|
| Kios API | `https://kiosapi.com/v1` | No (embedded zero-config key) |

`/models list` shows configured models; `/model` or Ctrl+T opens the selector.
The turn stays on its selected model during an outage; no automatic provider
or model failover is performed.

Config (model, effort, auto-approve) persists in
`~/.fullagent/config.json`; sessions save to `~/.fullagent/sessions/`.
If `~/.fullagent` is not writable, state transparently falls back to
`$TMPDIR/fullagent-<uid>` — the app never crashes on a read-only home.

## Layout

```
main.py            launcher — python main.py
fullagent/
  __init__.py      package
  config.py        providers, models, effort levels, paths
  systemprompt.py  the ONE home of every system prompt (single source)
  mastermind.py    prompt coherence: sealed vault, gate, composer, lineage
  tools.py         16 tools: files, shell, search, real-time web
  client.py        streaming OpenAI-compatible client (SSE, retries, cancel)
  agent.py         agent loop: LLM <-> tools, event-sourced on the kernel
  kernel.py        Temporal Kernel: append-only, content-addressed event log
  memory.py        episodic memory + dead-end ledger (fold-derived)
  goal.py          goal contracts with machine-checkable done-criteria
  judge.py         deterministic verification predicates (no LLM judging)
  roles.py         role capability catalog for deterministic planning
  workflows.py     saved multi-step pipelines — phased orchestration (serial steps)
  autopilot.py     self-routing: auto goal mode / real-time web
  router.py        task classification and routing through Union Alpha
  semantic.py      semantic vector memory — meaning-based recall
  speculate.py     speculative execution — prefetch read-only tool calls
  dashboard.py     live observability — real-time ledger projection
  daemon.py        mission control — resumable long-running missions
  healer.py        self-healing — root-cause capture, fix, retry, lesson
  skills.py        skill forge — self-authored, safety-gated tools
  council.py       adversarial debate — thesis/antithesis + blind judge
  taint.py         static analysis — taint flows, complexity, import cycles
  kgraph.py        knowledge graph — entities + typed relations, impact
  cov.py           real line coverage — sys.settrace measurement
  fuzz.py          property-based fuzzing — generators + crash shrinking
  mutate.py        mutation testing — AST mutants vs the test suite
  tui.py           persistent double-line box, overlays, streaming, approval
  __main__.py      entry point
```

The event log lives at `~/.fullagent/eventlog.jsonl` (override the directory
with `FULLAGENT_HOME`). Each module ships a self-test:
`python -m fullagent.kernel` (and `.memory`, `.goal`, `.judge`,
`.roles`, `.autopilot`, `.systemprompt`, `.mastermind`, `.router`,
`.semantic`, `.speculate`, `.dashboard`, `.daemon`, `.healer`, `.skills`,
`.council`, `.taint`, `.kgraph`, `.cov`, `.fuzz`, `.mutate`).

Streaming/UI regressions (no API key needed), from this directory:
`python -m unittest discover -s tests -v`.

## System prompts — one file, one delivery path

Every system prompt the model ever sees lives in **`fullagent/systemprompt.py`**
and nowhere else — the module imports it. Three structural guarantees:

- **Single source of truth.** Edit a prompt in `systemprompt.py` and it
  changes wherever the main agent requests are built.
- **One system message.** `systemprompt.with_system()` guarantees exactly
  one system message, at position 0. Mid-history "system" hints are removed
  because providers merge them inconsistently — the sealed prompt is the
  only system voice.
- **Wire-level integrity.** `PromptGate.seal_wire()` runs on the exact
  payload about to be sent (after pruning, budget capping and compaction
  retries). It re-seats a missing or tampered prompt and records the
  restore in the event log.
- **Live reload.** The vault watches `systemprompt.py`'s
  `(mtime_ns, size)`. Edit the file in a running session and the next
  `resolve()` reloads the module and re-seals changed prompts — no restart.
  Runtime-registered prompts survive the reload. A broken edit is logged
  as `prompt.reload_failed` and the last good prompt stays in force.

> **Source vs installed binary.** Editing `systemprompt.py` only affects a
> run started from this checkout (`python3 main.py`). The standalone release
> binary and a `pip install` carry their own bundled copy of `fullagent/`;
> they will not see edits in your clone. Run from source while iterating on
> prompts, or reinstall after editing.

Two prompts ship in the registry, switchable live with `/prompt`:

| Name | Size | What it is |
|---|---|---|
| `main` | ~5k chars | the compact sovereign-agent prompt (default) |
| `master` | `main` + optional spec | MAIN plus `fullagent/project.txt` when that optional file is present; otherwise it degrades to MAIN (no empty banner) |

The file also owns every non-conversational prompt: `subagent_prompt()` for
delegated workers, and `INTERNAL_RULES` plus task builders for subsystem
model calls (daemon steps, council/debate speakers, the fast dual path, the
work compiler and program synthesis). Those internal calls share one base
contract but keep their strict output formats — the compiler still returns
only JSON, the synthesizer only source.

Add more prompts later by dropping a constant in `systemprompt.py` and
registering it in the `PROMPTS` map (or call `register()` at runtime).

## Output budget — 200k tokens

Every effort level requests **200,000 output tokens** (`config.MAX_TOKENS`).
The client clamps requests to the Kios backend's **131,072-token** completion
ceiling and the remaining context window before sending them to Kios.

## Mastermind — coherence, not coercion

`fullagent/mastermind.py` makes the prompt the coherent center of every
request. It does not, and cannot, override a model's own safety training or
a provider's server-side policy — if a request is refused upstream, no
client-side prompt layout changes that. What it *does* guarantee is that the
prompt you edit is the prompt that is actually sent. Four deterministic
mechanisms:

| Mechanism | What it does |
|---|---|
| **PromptVault** | Every prompt is sealed with a sha256 fingerprint and recorded in the event log. Prompts registered at runtime are sealed on demand; a changed prompt is re-sealed — no stale copy is ever served. |
| **PromptGate** | The turn-start door to the model. `gate.dispatch()` guarantees `messages[0]` carries the sealed prompt byte-for-byte at the front, re-seats it if anything shadowed it, and seals a `prompt.dispatch` lineage event. |
| **Wire gate** | `gate.seal_wire()` runs on the exact payload at send time, after `prune_messages`, `bound_prompt`, prewarm and compaction rebuilds. It enforces one system message at index 0, removes every competing mid-history system message, and restores a dropped or tampered prompt. Restores are recorded as `prompt.wire_restore`; cleanups as `prompt.wire_clean`. |
| **CoherenceComposer** | Live context (constitution, goal, web mode, memory) is composed beneath the sealed prompt as one document. Each section is framed as *input to* the prompt, ordered by authority and deduplicated, so it cannot pose as a second policy. |

The agent loop no longer injects "[MANDATORY COMPLIANCE]" user messages or
mid-history system hints. Those compete with the real prompt and are
discounted by modern models; runtime guidance now travels in the user
channel, explicitly tagged as subordinate. Inspect the live ledger with
`/mastermind`.

## v3 — eight advanced subsystems

All eight are event-sourced on the same Temporal Kernel: every decision,
prediction, heal, skill and verdict is a sealed event, and every status
view is a pure fold. Nothing keeps private state, so nothing can drift
from the log.

| Module | What it does | Command |
|---|---|---|
| **router.py** | A deterministic difficulty classifier scores each task and records routing decisions. This build routes all tasks to Union Alpha; no alternate models or cost-saving switches are available. | `/router` |
| **semantic.py** | Hippocampus 2.0. Every episode, fact and dead-end is embedded via signed feature hashing (stdlib only, no numpy) and recalled by cosine similarity — "how did we solve a similar problem before?", including remembering what *failed*. The index is a pure projection of the log and refreshes itself. Recall is injected into the memory context section each turn. | `/recall <q>` |
| **speculate.py** | While the model thinks, the agent predicts the read-only calls it will likely make (paths in your message, search verbs, siblings of recent reads) and prefetches them in a background pool. When the model actually asks, the result is served from cache — a hit instead of an execution. Only whitelisted read-only tools can ever be prefetched; a speculative write is structurally impossible. | `/spec` |
| **dashboard.py** | The X-ray: cost, tokens, goal progress bar, council activity, routing spend, speculation hit-rate, memory counts, verdicts, loop alerts and budget events — one screen, always agreeing with the kernel because it is a fold. | `/dashboard` |
| **daemon.py** | Mission Control. A mission is a queue of steps advanced one tick at a time; every tick checkpoints, so a restart resumes from the last checkpoint (at most one in-flight tick is lost). A step that exhausts its retries BLOCKS the mission visibly — never silently skipped. Wake conditions are deterministic fold predicates. | `/mission` |
| **healer.py** | When a tool fails, the healer captures the error, classifies it against a root-cause taxonomy (16 patterns; unknown is honest, never guessed), and seals the lesson. With a fixer + recheck attached it runs the full loop: fix → re-run the original check → only a green re-run counts as healed. Every tool error in the agent loop is captured automatically. | `/heal` |
| **skills.py** | The self-evolving tool author. A new skill (Python function) passes four gates before it can run: parse → shape (entry fn + docstring) → safety (AST scan: no subprocess/eval/exec/forbidden imports/dunder access/globals) → its own shipped test cases. Passing skills persist to `~/.fullagent/skills/` and register as live tools; failures are sealed with the exact reason. | `/skills` |
| **council.py** | Adversarial debate for high-stakes calls: THESIS argues for, ANTITHESIS argues against and must attack the thesis's strongest point, then a BLIND judge sees only the two anonymised arguments (never the question's framing) and decides on argument strength alone. Verdicts carry winner, confidence and reason. | `/council <q>` |

## v4 — five professional engineering subsystems

Same discipline as v3: pure stdlib, deterministic, no model calls, every
result sealed into the Temporal Kernel as an event, every status view a
pure fold. These are real engineering tools, not estimates.

| Module | What it does | Command / Tool |
|---|---|---|
| **taint.py** | Real static analysis over the AST (not regex): taint tracking from declared sources (input, env, network, file reads) to sinks (eval, exec, subprocess, sql, writes) with the exact propagation path; cyclomatic complexity per function with hotspots; module-level import-cycle detection via iterative DFS. | `/analyze <path>` · tool `analyze_code` |
| **kgraph.py** | The knowledge graph: entities (modules, functions, classes, files, goals, episodes, facts) and typed relations (defines, calls, imports, touches, learned) built straight from the AST and the event fold. Queries are real graph operations — BFS reachability, reverse lookups, and impact sets ("what breaks if I change X?"). | `/graph [index\|query\|impact]` · tools `graph_index`, `graph_query`, `graph_impact` |
| **cov.py** | Genuine line coverage, not an estimate: `sys.settrace` (the same hook `coverage.py` uses) records every executed line of the target while a subject runs, compared against executable lines derived from the AST. The trace only records — never alters control flow — and is always restored. | `/coverage` · tool `measure_coverage` |
| **fuzz.py** | Property-based fuzzing: typed generators (int, str, list, dict, bytes) biased toward boundaries (0, -1, empty, huge, unicode), plus mutated inputs. Crashes are SHRUNK to a minimal reproducer — the difference between "it crashed somewhere" and "here is the smallest input that breaks it." Deterministic under a seed. | `/fuzz` · tool `fuzz_target` |
| **mutate.py** | Mutation testing — answers what tests alone cannot: *can your tests actually catch bugs?* AST NodeTransformers generate real mutants (operator flips, condition negations, broken returns); the suite runs against each. Killed = suite caught it; survived = a real hole. Score = killed / (killed + survived). The original file is always restored. | `/mutate <file> <suite-cmd>` |

---

<div align="center">

**FullAgent** — *Pure Python · Event-Sourced · Self-Healing*

`python main.py` → banner · `~/.fullagent/config.json` → persistence · `FULLAGENT_HOME` → override

</div>
