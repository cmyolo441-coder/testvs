// The OMH desktop half: the HUD the OMH TUI draws (src/tui/panels.py),
// redrawn for Hermes Desktop as a right-hand pane -- plan checklist, agent
// rows with their routed model and metrics, the fanout DAG while one is
// active -- and a compact status-bar item, in the app's own visual language
// rather than the terminal's.
//
// Plain ESM, loaded uncompiled through the app's disk-plugin door: only
// '@hermes/plugin-sdk', 'react' and 'react/jsx-runtime' resolve, and JSX
// syntax is a SyntaxError there, so the tree is written with jsx()/jsxs().
// The file is checked for the one character JSX needs, so every comparison
// below is written in its greater-than form.
//
// Tailwind never scans a disk plugin, so the pane's own classes live in a
// style element this file appends on register and removes on dispose. Every
// colour resolves through the app's theme tokens (var(--ui-*)), never a
// literal; the few core utility classes used are ones the app compiles for
// its own panes.
//
// Electron copies this file from plugins/omh/desktop/ to desktop-plugins/omh/
// beside a .hermes-package.json marker, and that marker keeps the half OFF
// until the user switches it on under Capabilities -> Plugins. Every figure
// shown comes from the sibling backend (dashboard/plugin_api.py) through
// ctx.rest('/hud'), which reaches /api/plugins/omh/hud inside the gateway
// process, plus the host's own session state for the main-session model
// line; nothing is computed here beyond formatting, so a number the reader
// did not produce is never displayed.
import {
  Badge,
  GlyphSpinner,
  host,
  PanelEmpty,
  PANES_AREA,
  Skeleton,
  STATUSBAR_AREAS,
  StatusDot,
  Tip,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

export const PLUGIN_ID = 'omh'
export const PANE_ID = 'hud'
export const POLL_MS = 5000
export const STYLE_ID = 'omh-desktop-pane-style'
export const PANE_WIDTH = '320px'
// The TUI's plan window: eight items anchored just before the first
// unfinished one, the rest folded into earlier/later counts.
export const PLAN_WINDOW_ROWS = 8
export const GRAPH_NODE_LIMIT = 8

// Bound in register(); ctx.rest is scoped to this plugin's namespace.
let rest = null

export function hudPath(sessionId) {
  return sessionId ? `/hud?session=${encodeURIComponent(sessionId)}` : '/hud'
}

function fetchHud(sessionId) {
  if (!rest) {
    return Promise.reject(new Error('omh plugin context is not registered'))
  }
  return rest(hudPath(sessionId))
}

// ---------------------------------------------------------------------------
// Formatters ported from the retired Hermes status widget, as the OMH TUI
// panels' are, so the surfaces spell one figure the same way: 77.0k, 4m 2s,
// ~$0.0431, category:architect(model:effort).
// ---------------------------------------------------------------------------

// Parentheses stay in the allowlist because the row identity is SHAPED by
// them: `category:architect(claude...)` and `turn 3 (12 tools)`.
// Same rule as the OMH TUI panels: whole escape sequences out,
// then letters, digits and ordinary punctuation in (`person's` must not become `persons`).
const ESCAPE_SEQUENCE = /\x1b\[[0-?]*[ -\/]*[@-~]?|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[ -\/]*[0-~]?/g
export const sanitizeText = value => String(value ?? '')
  .replace(ESCAPE_SEQUENCE, '')
  .replace(/[^\p{L}\p{N} .:/_·|+()\[\]!\-',;?"#%&=@~]/gu, '')
export const safeText = value => sanitizeText(value).slice(0, 96)
// The reader caps an action title at 140 characters; the row's tooltip is
// the one place that shows it whole, so it takes a longer cut than safeText.
export const longText = value => sanitizeText(value).slice(0, 160)
const plural = (count, noun) => `${count} ${noun}${count === 1 ? '' : 's'}`

// One decimal always kept above a thousand, bare integers under it; a
// recorded zero renders as `0`, not as nothing.
export function tokenCountText(value) {
  if (!Number.isFinite(value) || 0 > value) return ''
  if (1000 > value) return `${Math.floor(value)}`
  const [amount, unit] = 999_950 > value ? [value / 1000, 'k'] : [value / 1_000_000, 'm']
  return `${amount.toFixed(1)}${unit}`
}

export function elapsedText(value) {
  if (!Number.isFinite(value)) return ''
  const seconds = Math.max(0, Math.floor(value))
  if (60 > seconds) return `${seconds}s`
  if (3600 > seconds) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}m`
}

// An approximation keeps its `~`, a confirmed zero shows its provenance
// marker, an unvouched zero shows nothing.
export function costSegmentText(row) {
  if (!Number.isFinite(row.cost_usd)) return ''
  if (row.cost_usd > 0) return `${row.cost_approximate ? '~' : ''}$${row.cost_usd.toFixed(4)}`
  if (row.cost_approximate) return ''
  const provenance = safeText(row.cost_status) || safeText(row.cost_source)
  return provenance ? `$${row.cost_usd.toFixed(4)} (${provenance})` : ''
}

export function hudStateLabel(active, agents) {
  if (!active) return 'ready'
  const running = Number(agents.running) || 0
  const blocked = Number(agents.blocked) || 0
  const done = Number(agents.completed) || 0
  if (!running && !blocked && done) return `${done} done`
  const parts = [plural(Number(agents.active) || 0, 'agent')]
  if (running) parts.push(`${running} running`)
  if (blocked) parts.push(`${blocked} blocked`)
  if (done) parts.push(`${done} done`)
  return parts.join(' · ')
}

const MAESTRO_EXECUTOR_SHORT_NAMES = { codex: 'codex', claude_code: 'claude', pi_family_runtime: 'pi', hermes_local: 'hermes' }

// The route column shows WHICH model runs, not who serves it: the label
// keeps only the segment after the last slash. Shapes, as the TUI draws
// them: `category:architect(model:effort)`, `category:quick(model fallback)`,
// `category:deep(model inherit)`, `inherit(model:effort)`, a bare
// `model:effort`, `(codex/maestro model)` and `kanban/assignee(model:effort)`.
export function routeIdentity(row) {
  const modelName = safeText(row.model).split('/').pop()
  const model = [modelName, safeText(row.effort)].filter(Boolean).join(':')
  const dispatchLane = safeText(row.dispatch_lane)
  const dispatchExecutor = safeText(row.executor_profile)
  const dispatchIdentity = dispatchLane
    ? `(${MAESTRO_EXECUTOR_SHORT_NAMES[dispatchExecutor] || dispatchExecutor}/${dispatchLane}${modelName ? ` ${modelName}` : ''})`
    : ''
  const category = safeText(row.category)
  const routeOrigin = safeText(row.route_origin)
  const routeCategory = safeText(row.route_category)
  const routeTag = routeOrigin === 'fallback' ? 'fallback' : routeOrigin === 'exhausted_to_inherit' ? 'inherit' : ''
  const routeDetail = [model, routeTag].filter(Boolean).join(' ')
  const displayCategory = routeOrigin === 'exhausted_to_inherit' && routeCategory ? routeCategory : category
  const route =
    displayCategory === 'inherit'
      ? `inherit${model ? `(${model})` : ''}`
      : displayCategory
        ? `category:${displayCategory}${routeDetail ? `(${routeDetail})` : ''}`
        : model
  const routeKind = routeOrigin === 'fallback' || routeOrigin === 'exhausted_to_inherit' ? 'route-fallback' : 'route'
  if (safeText(row.lane_backend) === 'kanban') {
    const assignee = safeText(row.assignee) || 'unassigned'
    const effort = safeText(row.effort)
    const detail = modelName ? `${modelName}${effort ? `:${effort}` : ''}` : ''
    return { kind: 'kanban', text: `kanban/${assignee}${detail ? `(${detail})` : ''}` }
  }
  return dispatchLane ? { kind: 'maestro', text: dispatchIdentity } : { kind: routeKind, text: route }
}

// The identity never breaks inside a token: it is split where the TUI's
// grammar has a seam (after the `category:` literal, before the model
// parenthesis) and each piece is drawn nowrap with a break opportunity
// between them, so a narrow pane wraps `(model:effort)` under the lane
// instead of cutting `claude-fable-5-1` in half.
export function routeFragments(text) {
  const pieces = []
  let remainder = text
  if (remainder.startsWith('category:')) {
    pieces.push('category:')
    remainder = remainder.slice('category:'.length)
  }
  const paren = remainder.indexOf('(')
  if (paren > 0) {
    pieces.push(remainder.slice(0, paren))
    remainder = remainder.slice(paren)
  }
  if (remainder) pieces.push(remainder)
  return pieces
}

// The tag in front of the id names what runs the row: `bot` is a board
// worker, `sub` a delegate_task child of this session, `main` the Maestro
// binding row; a dispatched executor row carries no tag, as in the TUI.
export const kindTag = (row, main) =>
  main ? 'main' : safeText(row.lane_backend) === 'kanban' ? 'bot' : safeText(row.dispatch_lane) || safeText(row.executor_profile) ? '' : 'sub'

function sessionMetrics(payload) {
  const rows = []
    .concat(Array.isArray(payload.maestro?.rows) ? payload.maestro.rows : [])
    .concat(Array.isArray(payload.subagents?.rows) ? payload.subagents.rows : [])
  const tokens = rows.reduce((sum, row) => sum + (Number.isFinite(row.tokens) ? row.tokens : 0), 0)
  return { tokens: tokens > 0 ? `${tokenCountText(tokens)} tokens` : '' }
}

const skippedClause = counts => {
  const skipped = Number(counts.skipped) || 0
  return skipped > 0 ? ` (${skipped} skipped)` : ''
}
const workedCount = counts => Math.max(0, (Number(counts.done) || 0) - (Number(counts.skipped) || 0))

// The badge row: header segments the TUI draws after the state label, in
// the same priority, each only when the reader observed it.
export function chipList(payload) {
  const chips = []
  const repeat = payload.repeat || {}
  const repeatCount = Number(repeat.consecutive) || 0
  const repeatPeriod = Number(repeat.period) || 1
  const repeatStage = (Number(repeat.intercepted) || 0) >= 1 ? safeText(repeat.stage) : ''
  if (repeat.status === 'observed' && repeatCount >= 2) {
    chips.push({
      key: 'repeat',
      text: `repeat x${repeatCount}${repeatPeriod > 1 ? ` cycle-of-${repeatPeriod}` : ''}${repeatStage === 'blocking' ? ' blocked' : repeatStage === 'approval' ? ' approval' : ''}`,
      variant: repeatStage === 'approval' ? 'destructive' : repeatStage === 'blocking' ? 'warn' : 'muted'
    })
  }
  const board = payload.kanban || {}
  if ((Number(board.rows_total) || 0) > 0) {
    chips.push({
      key: 'board',
      text: `board ${Number(board.queued) || 0}q ${Number(board.running) || 0}r ${Number(board.blocked) || 0}b`,
      variant: 'muted'
    })
  }
  if (board.dispatcher_presence === 'not_observed') {
    chips.push({ key: 'dispatcher', text: 'dispatcher not observed', variant: 'warn' })
  }
  const activity = payload.activity || {}
  if (activity.live && activity.post_tool_call_observed) {
    chips.push({
      key: 'tools',
      text: `${plural(Number(activity.open_call_count) || 0, 'tool')} · ${elapsedText(activity.oldest_open_elapsed_seconds) || '0s'}`,
      variant: 'warn'
    })
  }
  const yolo = payload.yolo || {}
  if (yolo.status === 'observed') {
    chips.push({ key: 'yolo', text: `yolo ${yolo.enabled ? 'on' : 'off'}`, variant: yolo.enabled ? 'warn' : 'muted' })
  }
  const metrics = sessionMetrics(payload)
  if (metrics.tokens) chips.push({ key: 'tokens', text: metrics.tokens, variant: 'muted' })
  const shot = payload.parallel_shot || {}
  // No chip shows a number below 2: calls that never overlapped were no parallel shot (the TUI's `_shot_badge`).
  if (shot.status === 'observed') {
    const open = Number(shot.open_count) || 0
    const peak = Number(shot.peak_open_count) || 0
    if (open > 1) chips.push({ key: 'shot', text: `parallel shot ×${open}`, variant: 'default' })
    else if (peak > 1) {
      const observedAt = shot.observed_at ? Date.parse(shot.observed_at) : NaN
      const age = Number.isFinite(observedAt) ? elapsedText(Math.max(0, (Date.now() - observedAt) / 1000)) : ''
      chips.push({ key: 'shot', text: `parallel shot ×${peak}${age ? ` (${age} ago)` : ''}`, variant: 'muted' })
    }
  }
  return chips
}

// The TUI's plan window: `limit` items anchored one before the first
// unfinished item, so current work is always on screen.
export function planWindow(items, limit = PLAN_WINDOW_ROWS) {
  const total = items.length
  const firstRemaining = items.findIndex(item => item.state !== 'done')
  const anchor = 0 > firstRemaining ? 0 : Math.max(0, firstRemaining - 1)
  const start = total > limit ? Math.min(anchor, total - limit) : 0
  const end = Math.min(total, start + limit)
  return { start, end, earlier: start, later: total - end }
}

const depthOf = item => {
  const depth = Number(item.depth)
  return Number.isInteger(depth) && depth > 0 ? Math.min(depth, 3) : 0
}

// Consecutive items of one phase form a group; the phase is drawn once, on
// the group's first item. A subtask with no phase continues its parent's.
export function planGroups(items) {
  const groups = []
  for (const item of items) {
    const phase = safeText(item.phase)
    const last = groups[groups.length - 1]
    if (last && (last.phase === phase || (!phase && depthOf(item) > 0))) last.items.push(item)
    else groups.push({ phase, items: [item] })
  }
  return groups
}

const REASON_CHARS = 48 // TODO_BLOCKED_REASON_DISPLAY_CHARS, runtime_reader.py
const reasonOf = item => {
  const reason = safeText(item.blocked_reason)
  return reason.length > REASON_CHARS ? `${reason.slice(0, REASON_CHARS - 1)}…` : reason
}

// What the pane and the status item render from one poll: the payload when
// the reader answered, otherwise the one state the surface must say. A
// transport failure keeps the last payload on screen under a quiet notice.
export function paneModel({ gateway, data, error }) {
  if (gateway !== 'open') {
    return { payload: null, status: `gateway ${gateway}`, notice: '', loading: false }
  }
  if (data && data.error) {
    return { payload: null, status: 'reader error', notice: `omh backend: ${data.error}`, loading: false }
  }
  if (error) {
    const message = error.message || String(error)
    return { payload: data || null, status: data ? '' : 'unavailable', notice: `omh backend: ${message}`, loading: false }
  }
  if (!data) {
    return { payload: null, status: '', notice: '', loading: true }
  }
  return { payload: data, status: '', notice: '', loading: false }
}

export function paneVersion(payload) {
  const version = safeText(payload.version)
  if (version && version !== 'unknown') return version
  const bundle = payload.plugin && safeText(payload.plugin.version)
  return bundle && bundle !== 'unknown' ? bundle : 'dev'
}

const globalPrefix = payload => {
  const agents = payload.subagents || {}
  const maestro = payload.maestro || {}
  const maestroGlobal = Array.isArray(maestro.rows) && maestro.rows.some(row => row.scope === 'global')
  return agents.scope === 'global' || agents.scope === 'mixed' || maestroGlobal ? '[global] ' : ''
}

export function headerLabel(payload) {
  return `${globalPrefix(payload)}${hudStateLabel(!!payload.active, payload.subagents || {})}`
}

// The status-bar item: one glyph and a few short tokens; the detail rides
// the tooltip. Nothing at all while the reader has not answered and nothing
// failed, the way the Kanban count stays out of the bar when idle.
export function statusItem({ gateway, data, error }) {
  if (gateway !== 'open') return null
  if (data && data.error) {
    return { segments: [{ text: 'omh error', tone: 'muted' }], detail: `omh backend: ${data.error}` }
  }
  if (!data) {
    return error ? { segments: [{ text: 'omh unavailable', tone: 'muted' }], detail: `omh backend: ${error.message || String(error)}` } : null
  }
  const active = !!data.active
  const agents = data.subagents || {}
  const segments = []
  if (active) {
    const running = Number(agents.running) || 0
    const blocked = Number(agents.blocked) || 0
    const done = Number(agents.completed) || 0
    if (!running && !blocked && done) segments.push({ text: `${done} done`, tone: 'tertiary' })
    else segments.push({ text: plural(Number(agents.active) || 0, 'agent'), tone: 'tertiary' })
    if (running) segments.push({ text: `${running} running`, tone: 'ok' })
    if (blocked) segments.push({ text: `${blocked} blocked`, tone: 'error' })
  } else {
    segments.push({ text: 'ready', tone: 'muted' })
  }
  const todo = data.todo || {}
  const counts = todo.counts || {}
  const detail = [headerLabel(data)]
  if (todo.status === 'established') {
    segments.push({ text: `${counts.done ?? 0}/${counts.total ?? 0}`, tone: 'warn' })
    detail.push(`plan ${safeText(todo.title) || 'untitled'} ${counts.done ?? 0}/${counts.total ?? 0}`)
  } else if (todo.status === 'all_done') {
    segments.push({ text: `✓ ${workedCount(counts)}/${counts.total ?? 0}`, tone: 'ok' })
    detail.push(`plan ${safeText(todo.title) || 'untitled'} finished`)
  }
  const line = data.display && typeof data.display.line === 'string' ? data.display.line : ''
  if (line) detail.push(line)
  return { segments, detail: detail.join(' · ') }
}

// ---------------------------------------------------------------------------
// Styles. Prefixed classes only; tokens, not literals. Mono cells name the
// app's bundled 'JetBrains Mono' first (styles.css @font-face), then the
// theme's mono stack, so ids, routes and figures share the transcript's face.
// One DOM serves both pane widths: up to the breakpoint an agent row stacks
// its three lines; past it the row's cells join one aligned grid (the line
// wrappers become `display: contents`) under a row of column labels.
// ---------------------------------------------------------------------------

const MONO = "'JetBrains Mono',var(--font-mono,var(--dt-font-mono,monospace))"

export const PANE_CSS = `
.omh-pane{display:flex;flex-direction:column;height:100%;min-width:0;container-type:inline-size;container-name:omh-pane;font-family:var(--font-sans,var(--dt-font-sans,sans-serif));font-size:12px;line-height:1.35;color:var(--ui-text-primary)}
.omh-pane-mono{font-family:${MONO};font-variant-numeric:tabular-nums}
.omh-pane-head{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 10px 3px;min-width:0}
.omh-pane-brand{display:flex;align-items:center;gap:6px;min-width:0;font-weight:600;white-space:nowrap}
.omh-pane-state{display:flex;align-items:center;gap:5px;min-width:0;font-size:11px;color:var(--ui-text-tertiary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.omh-pane-state span{overflow:hidden;text-overflow:ellipsis}
.omh-pane-session{display:flex;flex-wrap:wrap;align-items:baseline;padding:0 10px 3px;font-size:11px;color:var(--ui-text-quaternary);min-width:0}
.omh-pane-chips{display:flex;flex-wrap:wrap;gap:4px;padding:3px 10px 4px}
.omh-pane-metric{white-space:nowrap}
.omh-pane-metric+.omh-pane-metric::before{content:'·';margin:0 4px;color:var(--ui-text-quaternary)}
.omh-pane-rule{flex:none;border-top:1px solid var(--ui-stroke-tertiary);margin-top:5px}
.omh-pane-body{min-height:0;flex:1;overflow-y:auto;overscroll-behavior:contain}
.omh-pane-section{display:flex;align-items:baseline;gap:8px;padding:8px 10px 3px;min-width:0}
.omh-pane-section-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--ui-text-quaternary);white-space:nowrap}
.omh-pane-section-title{min-width:0;flex:1;font-size:12px;color:var(--ui-text-primary);overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.omh-pane-section-count{margin-left:auto;font-size:11px;font-variant-numeric:tabular-nums;color:var(--ui-text-quaternary);white-space:nowrap}
.omh-pane-section[data-column=plan]{max-width:640px;box-sizing:border-box}
.omh-pane-plan{padding:0 10px 4px;max-width:640px;box-sizing:border-box;min-width:0}
.omh-pane-bar{height:2px;margin:1px 0 6px;border-radius:1px;background:var(--ui-bg-tertiary);overflow:hidden}
.omh-pane-bar-fill{height:100%;background:var(--ui-accent)}
.omh-pane-bar-fill[data-done=true]{background:var(--ui-success,var(--ui-green))}
.omh-pane-phase{display:flex;align-items:baseline;gap:8px;margin-top:5px;font-size:12px;font-weight:600;color:var(--ui-text-primary);min-width:0}
.omh-pane-phase[data-current=true]{color:var(--ui-cyan)}
.omh-pane-phase-name{min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.omh-pane-phase-count{margin-left:auto;font-size:11px;font-weight:400;font-variant-numeric:tabular-nums;color:var(--ui-text-quaternary)}
.omh-pane-phase-items{margin:2px 0 0 12px;padding-left:8px;border-left:2px solid var(--ui-stroke-tertiary)}
.omh-pane-plan-items{margin-top:2px}
.omh-pane-item{display:grid;grid-template-columns:14px minmax(0,1fr);gap:0 4px;align-items:baseline;min-width:0;padding:1px 0}
.omh-pane-item-glyph{text-align:center;font-family:${MONO};color:var(--ui-text-tertiary)}
.omh-pane-item-text{min-width:0;overflow-wrap:anywhere}
.omh-pane-item-label{color:var(--ui-text-primary)}
.omh-pane-item[data-state=done] .omh-pane-item-label{color:var(--ui-text-quaternary);text-decoration:line-through}
.omh-pane-item[data-state=active] .omh-pane-item-label{font-weight:600}
.omh-pane-item-note{display:block;margin-top:1px;padding-left:6px;font-size:11px;color:var(--ui-text-quaternary);overflow-wrap:anywhere}
.omh-pane-fold{padding:1px 0;font-size:11px;color:var(--ui-text-quaternary)}
.omh-pane-agents{padding:0 10px 8px;min-width:0}
.omh-agent-head{display:none}
.omh-agent+.omh-agent{margin-top:8px}
.omh-agent-l1{display:flex;align-items:baseline;gap:6px;min-width:0}
.omh-agent-l2,.omh-agent-l3{display:flex;flex-wrap:wrap;align-items:baseline;padding-left:20px;font-size:11px;min-width:0}
.omh-agent-l3{color:var(--ui-text-quaternary)}
.omh-agent-cell{min-width:0;white-space:nowrap}
.omh-agent-l2 .omh-agent-cell:empty,.omh-agent-l3 .omh-agent-cell:empty{display:none}
.omh-agent-sep{margin:0 4px 0 0;padding-left:4px;color:var(--ui-text-quaternary)}
.omh-agent-glyph{flex:0 0 14px;width:14px;text-align:center;font-family:${MONO}}
.omh-agent-tag,.omh-agent-id{font-family:${MONO};font-size:11px;color:var(--ui-text-quaternary)}
.omh-agent[data-lane=kanban] .omh-agent-tag{color:var(--ui-accent)}
.omh-agent-title{flex:1;overflow:hidden;text-overflow:ellipsis;color:var(--ui-text-primary)}
.omh-agent-route{font-family:${MONO};font-variant-numeric:tabular-nums}
.omh-agent-route span{white-space:nowrap}
.omh-agent-figure{font-family:${MONO};font-variant-numeric:tabular-nums}
.omh-agent-more{padding-top:6px;font-size:11px;color:var(--ui-text-quaternary)}
@container omh-pane (min-width:481px){
.omh-pane-agents{display:grid;grid-template-columns:14px 34px 68px minmax(100px,1fr) minmax(0,max-content) 64px 60px 64px 64px;grid-auto-rows:26px;align-items:center}
.omh-agent-head,.omh-agent,.omh-agent-l1,.omh-agent-l2,.omh-agent-l3{display:contents}
.omh-agent-head span{display:flex;align-items:center;height:100%;min-width:0;padding-right:8px;overflow:hidden;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--ui-text-quaternary);white-space:nowrap}
.omh-agent-cell{display:flex;align-items:center;height:100%;padding-right:8px;box-sizing:border-box;overflow:hidden;font-size:11px}
.omh-agent-l2 .omh-agent-cell:empty,.omh-agent-l3 .omh-agent-cell:empty{display:flex}
.omh-agent-sep{display:none}
.omh-agent-title{font-size:12px;display:block;line-height:26px;white-space:nowrap}
.omh-agent-route{display:block;line-height:26px;white-space:nowrap;text-overflow:ellipsis}
.omh-agent-route wbr{display:none}
.omh-agent[data-alt=true] .omh-agent-cell{background:color-mix(in srgb,var(--ui-row-hover-background) 40%,transparent)}
.omh-agent-more{grid-column:1/-1;padding-top:0}
}
.omh-pane-graph{display:grid;gap:1px;padding:4px 10px 8px;font-family:${MONO};font-size:11px}
.omh-pane-graph-node{display:grid;grid-template-columns:1.6rem minmax(0,1fr);gap:0 4px;min-width:0}
.omh-pane-graph-node span:last-child{overflow-wrap:anywhere}
.omh-pane-notice{margin:6px 10px 8px;border-radius:6px;background:var(--chrome-action-hover);padding:6px 8px;font-size:11px;color:var(--ui-text-tertiary);overflow-wrap:anywhere}
.omh-pane-skeleton{height:10px;margin:6px 10px}
.omh-bar{display:inline-flex;height:100%;align-items:center;gap:4px;border-radius:0;padding:0 6px;font-size:.6875rem;color:var(--ui-text-tertiary);font-variant-numeric:tabular-nums;white-space:nowrap;transition:background-color .15s,color .15s}
.omh-bar:hover{background:var(--chrome-action-hover);color:var(--ui-text-primary)}
.omh-bar-seg+.omh-bar-seg::before{content:'·';margin:0 3px;color:var(--ui-text-quaternary)}
.omh-bar-glyph{font-weight:600;color:var(--ui-text-secondary)}
/* Tone classes last: a tone on a cell wins over the cell's own resting colour. */
.omh-pane-ok{color:var(--ui-success,var(--ui-green))}
.omh-pane-warn{color:var(--ui-yellow)}
.omh-pane-error{color:var(--ui-red)}
.omh-pane-label{color:var(--ui-cyan)}
.omh-pane-accent{color:var(--ui-accent)}
.omh-pane-muted{color:var(--ui-text-quaternary)}
.omh-pane-tertiary{color:var(--ui-text-tertiary)}
.omh-pane-primary{color:var(--ui-text-primary)}
`

// Appends the pane's stylesheet once and returns its remover. A host with no
// document (the node harness) gets a no-op.
export function injectStyle() {
  if (typeof document === 'undefined') return () => {}
  const existing = document.getElementById(STYLE_ID)
  if (existing) existing.remove()
  const style = document.createElement('style')
  style.id = STYLE_ID
  style.textContent = PANE_CSS
  document.head.append(style)
  return () => style.remove()
}

// ---------------------------------------------------------------------------
// Tree helpers. `node(type, props, children, key)` filters empty children
// and picks jsx/jsxs the way a compiler would.
// ---------------------------------------------------------------------------

const present = child => child !== null && child !== undefined && child !== false && child !== ''
function node(type, props, children, key) {
  const list = (Array.isArray(children) ? children : [children]).filter(present)
  if (list.length === 1) return jsx(type, { ...props, children: list[0] }, key)
  return jsxs(type, { ...props, children: list }, key)
}
const span = (className, children, key) => node('span', { className }, children, key)
const div = (className, children, key) => node('div', { className }, children, key)
const toneClass = tone => (tone ? `omh-pane-${tone}` : '')

function useHud() {
  const gateway = useValue(host.state.gateway)
  const sessionId = useValue(host.state.focusedStoredSessionId)
  const query = useQuery({
    queryKey: [PLUGIN_ID, 'hud', sessionId || ''],
    queryFn: () => fetchHud(sessionId),
    refetchInterval: POLL_MS,
    enabled: gateway === 'open',
    // Keep the last answer on screen across a session switch and while a
    // refetch is in flight, so the pane never blanks between polls.
    placeholderData: previous => previous
  })
  return {
    gateway,
    sessionId,
    data: query.data,
    error: query.error,
    receivedAt: Number.isFinite(query.dataUpdatedAt) ? query.dataUpdatedAt : 0
  }
}

// A one-second clock, ticking only while a running row exists so its elapsed
// time moves between polls; disposed with the component.
function useClock(running) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!running) return undefined
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [running])
  return now
}

// ---------------------------------------------------------------------------
// Pane sections
// ---------------------------------------------------------------------------

function Header({ model }) {
  const payload = model.payload
  const active = !!(payload && payload.active)
  const tone = !payload ? 'muted' : active ? 'warn' : 'good'
  const prefix = payload ? globalPrefix(payload).trim() : ''
  const label = payload ? hudStateLabel(active, payload.subagents || {}) : model.status
  return div('omh-pane-head', [
    div('omh-pane-brand', [
      span('', '⚚ OMH'),
      payload ? jsx(Badge, { variant: 'muted', size: 'xs', className: 'omh-pane-mono', children: `v${paneVersion(payload)}` }) : null
    ]),
    div('omh-pane-state', [jsx(StatusDot, { tone }), prefix ? span('omh-pane-muted', prefix) : null, label ? span('', label) : null])
  ])
}

// The main session's own model and usage come from the host, not the HUD:
// the payload names a model only per subagent row, and the TUI leaves the
// main model to the host statusline.
function SessionLine({ sessionId, model, usage }) {
  if (!sessionId) return div('omh-pane-session', span('omh-pane-metric', 'no focused session'))
  const fullModel = safeText(model)
  const name = fullModel.split('/').pop()
  const stats = usage && typeof usage === 'object' ? usage : {}
  return div('omh-pane-session', [
    name
      ? jsx(Tip, {
          label: fullModel,
          children: span('omh-pane-metric omh-pane-mono omh-pane-label', name)
        })
      : span('omh-pane-metric', 'model unknown'),
    Number.isFinite(stats.context_percent) ? span('omh-pane-metric omh-pane-mono', `ctx ${Math.round(stats.context_percent)}%`) : null,
    Number.isFinite(stats.total) && stats.total > 0 ? span('omh-pane-metric omh-pane-mono', `${tokenCountText(stats.total)} tok`) : null,
    Number.isFinite(stats.cost_usd) && stats.cost_usd > 0 ? span('omh-pane-metric omh-pane-mono', `$${stats.cost_usd.toFixed(3)}`) : null
  ])
}

function Chips({ payload }) {
  const chips = chipList(payload)
  if (!chips.length) return null
  return div(
    'omh-pane-chips',
    chips.map(chip => jsx(Badge, { variant: chip.variant, size: 'xs', className: 'omh-pane-mono', children: chip.text }, chip.key))
  )
}

// `PLAN  Desktop check          2/6` and `AGENTS            3`: the small label,
// an optional title in the body colour, the count at the right edge.
function SectionHeader({ label, title, count, countClass, column }) {
  return node('div', { className: 'omh-pane-section', 'data-column': column || 'full' }, [
    span('omh-pane-section-label', label),
    title ? span('omh-pane-section-title', title) : null,
    count ? span(`omh-pane-section-count ${countClass || ''}`, count) : null
  ])
}

function PlanItem({ item, live, unchanged }) {
  const state = item.state === 'active' || item.state === 'done' ? item.state : 'pending'
  const glyph = state === 'active' ? '●' : state === 'done' ? '✓' : '○'
  const glyphTone = state === 'active' ? (live ? 'ok' : 'warn') : state === 'done' ? 'muted' : 'tertiary'
  const reason = reasonOf(item)
  const depth = depthOf(item)
  const notes = []
  // Each note is its own line under the text, indented past the text start,
  // and a sibling of the label: a text decoration propagates to descendants
  // and cannot be undone below, so the strike lives on the label alone.
  if (reason) notes.push(div('omh-pane-item-note omh-pane-warn', `${state === 'done' ? 'skipped' : 'waiting'}: ${reason}`, 'reason'))
  if (state === 'active' && !live && unchanged) notes.push(div('omh-pane-item-note', `unchanged ${unchanged}`, 'unchanged'))
  return node(
    'div',
    { className: 'omh-pane-item', 'data-state': state },
    [
      span(`omh-pane-item-glyph ${toneClass(glyphTone)}`, glyph),
      node('span', { className: 'omh-pane-item-text', style: depth ? { paddingLeft: `${depth * 0.6}rem` } : undefined }, [span('omh-pane-item-label', safeText(item.text), 'label'), ...notes])
    ]
  )
}

// A plan whose steps Hermes' own `todo_list` holds (`native_todo`, the
// reader's `source`; `NATIVE_TODO_SOURCE` in native_todo.py). Hermes Desktop
// already draws that list in the composer's status stack -- a live Todos
// group while the turn runs, and on origin/main a "previous todos" disclosure
// after it (apps/desktop/src/app/chat/composer/status-stack/index.tsx, at
// v2026.9.7 and origin/main). Drawing the same items in this pane was the
// second checklist, so the section keeps only what Hermes cannot show: the
// open items carrying a reason omh_todo recorded, and the reader's stall
// finding. With neither, there is no PLAN section at all.
export const NATIVE_TODO_SOURCE = 'todo_list'

function NativeHeldPlanSection({ payload, todo }) {
  const items = Array.isArray(todo.items) ? todo.items : []
  const waiting = items.filter(item => item.state !== 'done' && reasonOf(item))
  const unchanged =
    todo.status === 'established' && todo.stall && todo.stall.status === 'unchanged' && Number.isFinite(todo.updated_age_seconds)
      ? elapsedText(Math.max(0, todo.updated_age_seconds))
      : ''
  if (!waiting.length && !unchanged) return []
  const activity = payload.activity || {}
  const live = activity.post_tool_call_observed ? !!activity.live : true
  const rows = waiting.map((item, index) => jsx(PlanItem, { item, live, unchanged }, `native-${index}`))
  if (unchanged && !waiting.some(item => item.state === 'active')) rows.push(div('omh-pane-fold', `unchanged ${unchanged}`, 'native-unchanged'))
  return [
    jsx(
      SectionHeader,
      { label: 'PLAN', title: 'in todo_list', count: waiting.length ? `${waiting.length} waiting` : '', countClass: 'omh-pane-warn', column: 'plan' },
      'plan-head'
    ),
    div('omh-pane-plan', rows, 'plan-body')
  ]
}

function PlanSection({ payload }) {
  const todo = payload.todo || {}
  const declared = todo.status === 'established' || todo.status === 'all_done'
  if (!declared) {
    return [
      jsx(SectionHeader, { label: 'PLAN', column: 'plan' }, 'plan-head'),
      jsx(PanelEmpty, { icon: 'checklist', title: 'No plan for this session', description: 'omh_todo declares one when a workflow starts.' }, 'plan-empty')
    ]
  }
  if (todo.source === NATIVE_TODO_SOURCE) return NativeHeldPlanSection({ payload, todo })
  const counts = todo.counts || {}
  const total = Number(counts.total) || 0
  const done = Number(counts.done) || 0
  const allDone = todo.status === 'all_done'
  const count = allDone ? `✓ ${workedCount(counts)}/${total}${skippedClause(counts)}` : `${done}/${total}${skippedClause(counts)}`
  const items = Array.isArray(todo.items) ? todo.items : []
  const window = planWindow(items)
  const shown = items.slice(window.start, window.end)
  const groups = planGroups(shown)
  const currentPhase = safeText(todo.display_phase)
  const activity = payload.activity || {}
  const live = activity.post_tool_call_observed ? !!activity.live : true
  const unchanged =
    todo.stall && todo.stall.status === 'unchanged' && Number.isFinite(todo.updated_age_seconds)
      ? elapsedText(Math.max(0, todo.updated_age_seconds))
      : ''
  const rows = []
  if (window.earlier) rows.push(div('omh-pane-fold', `… ${plural(window.earlier, 'earlier task')}`, 'earlier'))
  groups.forEach((group, groupIndex) => {
    const itemNodes = group.items.map((item, index) => jsx(PlanItem, { item, live, unchanged }, `${groupIndex}-${index}`))
    if (!group.phase) {
      rows.push(div('omh-pane-plan-items', itemNodes, `${groupIndex}-items`))
      return
    }
    // Every phase is its own header row with its own done/total, and its
    // items hang under it behind a left rule: the structure the TUI keeps,
    // drawn as an outline rather than a column.
    const phaseDone = group.items.filter(item => item.state === 'done').length
    rows.push(
      node(
        'div',
        { className: 'omh-pane-phase', 'data-current': group.phase === currentPhase ? 'true' : 'false' },
        [span('omh-pane-phase-name', group.phase), span('omh-pane-phase-count', `${phaseDone}/${group.items.length}`)],
        `${groupIndex}-phase`
      )
    )
    rows.push(div('omh-pane-phase-items', itemNodes, `${groupIndex}-items`))
  })
  if (window.later) rows.push(div('omh-pane-fold', `… ${plural(window.later, 'later task')}`, 'later'))
  return [
    jsx(SectionHeader, { label: 'PLAN', title: safeText(todo.title), count, countClass: allDone ? 'omh-pane-ok' : 'omh-pane-warn', column: 'plan' }, 'plan-head'),
    div(
      'omh-pane-plan',
      [
        node(
          'div',
          { className: 'omh-pane-bar' },
          node('div', {
            className: 'omh-pane-bar-fill',
            'data-done': allDone ? 'true' : 'false',
            style: { width: `${total ? Math.min(100, Math.round((done / total) * 100)) : 0}%` }
          }),
          'plan-bar'
        ),
        ...rows
      ],
      'plan-body'
    )
  ]
}

const STATE_TONE = { running: 'ok', done: 'muted', blocked: 'error', failed: 'error', stale: 'warn', queued: 'muted' }

function AgentGlyph({ row }) {
  const state = safeText(row.state) || 'running'
  const board = safeText(row.lane_backend) === 'kanban'
  const cls = 'omh-agent-glyph omh-agent-cell'
  if (state === 'blocked' || state === 'failed') return span(`${cls} omh-pane-error`, '▲')
  if (state === 'done') return span(`${cls} omh-pane-ok`, '✓')
  if (state === 'queued') return span(`${cls} omh-pane-muted`, '·')
  if (state === 'stale') return span(`${cls} omh-pane-warn`, '!')
  return span(`${cls} ${board ? 'omh-pane-accent' : 'omh-pane-warn'}`, jsx(GlyphSpinner, { spinner: 'braille', ariaLabel: 'running' }))
}

// An absolute path in an action title spends the whole line on directories
// nobody reads; the title keeps the path's last two segments behind `…/`,
// and the tooltip keeps the full sentence.
export function collapsePaths(text) {
  return String(text)
    .split(' ')
    .map(token => {
      if (!token.startsWith('/') || 25 > token.length) return token
      const segments = token.split('/').filter(Boolean)
      return segments.length > 1 ? `…/${segments.slice(-2).join('/')}` : token
    })
    .join(' ')
}

// The figures a row draws -- state, elapsed, tokens, cost -- and the ones it
// keeps for the tooltip (fallback, rate, cache, ctx, turn), from the same
// row fields the TUI reads.
export function rowFigures(row, extraSeconds) {
  const board = safeText(row.lane_backend) === 'kanban'
  const verdict = safeText(row.state) || 'running'
  const state = board ? safeText(row.native_status) || verdict : verdict
  const stateTone = verdict === 'blocked' || verdict === 'failed' ? 'error' : verdict === 'stale' ? 'warn' : board ? 'accent' : STATE_TONE[verdict] || 'ok'
  const running = !row.state || row.state === 'running'
  const elapsedSeconds = running ? (row.elapsed_seconds || 0) + extraSeconds : row.elapsed_seconds
  const elapsed = verdict === 'queued' ? '' : elapsedText(elapsedSeconds) || '0s'
  const tokens = tokenCountText(row.tokens)
  const extras = []
  if (Number.isFinite(row.fallback_count) && row.fallback_count > 0) extras.push(`fallback:${row.fallback_count}`)
  if (Number.isFinite(row.tokens_per_second)) extras.push(`${Math.round(row.tokens_per_second)} tok/s`)
  if (Number.isFinite(row.cache_hit_percentage)) extras.push(`cache ${row.cache_hit_percentage}%`)
  if (Number.isFinite(row.context_percentage)) extras.push(`ctx ${row.context_percentage}%`)
  const turn = Number.isFinite(row.turn_count) ? `turn ${row.turn_count}` : ''
  const tools = Number.isFinite(row.tool_count) ? `${row.tool_count} tools` : ''
  const turnTools = turn && tools ? `${turn} (${tools})` : turn || tools
  if (turnTools) extras.push(turnTools)
  return { state, stateTone, elapsed, tokens: tokens ? `${tokens} tok` : '', cost: costSegmentText(row), extras }
}

const ROUTE_TONE = { route: 'label', 'route-fallback': 'warn', maestro: 'warn', kanban: 'accent' }

// Every cell is rendered (the wide grid needs the placeholder). A dot leads
// a cell only when an earlier cell on the line carries a value, and it
// lives INSIDE that nowrap cell, so a stacked line neither opens with a
// separator nor ends on one; the wide grid hides the dots.
function dotted(cells) {
  let pending = false
  return cells.map(([className, children, filled], index) => {
    const leading = filled && pending ? [span('omh-agent-sep', '·', 'sep')] : []
    if (filled) pending = true
    return span(className, [...leading, ...(Array.isArray(children) ? children : [children])], `cell-${index}`)
  })
}
const AGENT_COLUMNS = ['', 'kind', 'id', 'task', 'route', 'state', 'elapsed', 'tokens', 'cost']

function AgentRow({ row, main, extraSeconds, alt }) {
  const board = safeText(row.lane_backend) === 'kanban'
  const tag = kindTag(row, main)
  const taskId = (safeText(row.task_id) || safeText(row.role) || 'agent').slice(0, 8)
  const action = longText(row.action)
  const route = routeIdentity(row)
  const routeChildren = []
  routeFragments(route.text).forEach((fragment, index) => {
    if (index > 0) routeChildren.push(jsx('wbr', {}, `wbr-${index}`))
    routeChildren.push(span('', fragment, `frag-${index}`))
  })
  const figures = rowFigures(row, extraSeconds)
  const tip = [action, route.text, ...figures.extras].filter(Boolean).join(' · ')
  return node(
    'div',
    { className: 'omh-agent', 'data-lane': board ? 'kanban' : 'native', 'data-alt': alt ? 'true' : 'false' },
    [
      div('omh-agent-l1', [
        jsx(AgentGlyph, { row }),
        span('omh-agent-tag omh-agent-cell', tag),
        span('omh-agent-id omh-agent-cell', taskId),
        action
          ? jsx(Tip, { label: tip, children: span('omh-agent-title omh-agent-cell', collapsePaths(action)) })
          : span('omh-agent-title omh-agent-cell', '')
      ]),
      div(
        'omh-agent-l2',
        dotted([
          [`omh-agent-route omh-agent-cell ${toneClass(ROUTE_TONE[route.kind])}`, routeChildren, !!route.text],
          [`omh-agent-cell ${toneClass(figures.stateTone)}`, figures.state, !!figures.state]
        ])
      ),
      div(
        'omh-agent-l3',
        dotted([
          ['omh-agent-figure omh-agent-cell', figures.elapsed, !!figures.elapsed],
          ['omh-agent-figure omh-agent-cell', figures.tokens, !!figures.tokens],
          ['omh-agent-figure omh-agent-cell', figures.cost, !!figures.cost]
        ])
      )
    ]
  )
}

function AgentsSection({ payload, receivedAt }) {
  const active = !!payload.active
  const maestro = payload.maestro || {}
  const agents = payload.subagents || {}
  const mainRows = active && Array.isArray(maestro.rows) ? maestro.rows.slice(0, 1) : []
  const rows = active && Array.isArray(agents.rows) ? agents.rows : []
  const anyRunning = [...mainRows, ...rows].some(row => !row.state || row.state === 'running')
  const now = useClock(anyRunning)
  if (!mainRows.length && !rows.length) return null
  const hidden = active ? Number(agents.hidden_rows) || 0 : 0
  const extraSeconds = receivedAt ? Math.max(0, (now - receivedAt) / 1000) : 0
  const all = [...mainRows.map(row => ({ row, main: true })), ...rows.map(row => ({ row, main: false }))]
  return [
    div('omh-pane-rule', [], 'agents-rule'),
    jsx(SectionHeader, { label: 'AGENTS', count: `${all.length + hidden}` }, 'agents-head'),
    div(
      'omh-pane-agents',
      [
        div('omh-agent-head', AGENT_COLUMNS.map((label, index) => node('span', {}, label, `col-${index}`)), 'agents-columns'),
        ...all.map((entry, index) =>
          jsx(AgentRow, { row: entry.row, main: entry.main, extraSeconds, alt: index % 2 === 1 }, `${entry.main ? 'main' : safeText(entry.row.task_id)}-${index}`)
        ),
        hidden ? div('omh-agent-more', `+${hidden} more`, 'agents-more') : null
      ],
      'agents-rows'
    )
  ]
}

const GRAPH_SUCCESS = new Set(['completed', 'already_completed', 'dry_run_planned'])
const GRAPH_FAILED = new Set([
  'capability_snapshot_invalid',
  'modality_unknown',
  'modality_unsupported',
  'modality_transformation_unobserved',
  'failed',
  'blocked',
  'blocked_by_dependency',
  'executor_not_ready',
  'unsupported_for_local_dispatch',
  'worktree_failed',
  'not_selected',
  'interrupted',
  'model_choice_required'
])

export function graphNodeView(nodeRecord) {
  const blockedBy = Array.isArray(nodeRecord.blocked_by) ? nodeRecord.blocked_by : []
  const dispatchState = safeText(nodeRecord.state) || 'unknown'
  const stuckState = safeText(nodeRecord.unit_state)
  const state = stuckState || dispatchState
  const stallSeconds = Number(nodeRecord.stalled_for_seconds) || 0
  const stuckReason = stuckState
    ? [safeText(nodeRecord.state_reason), stallSeconds > 0 ? `${elapsedText(stallSeconds)} since new output` : ''].filter(Boolean).join(', ')
    : ''
  const marker = stuckState ? '[~]' : nodeRecord.in_frontier ? '[R]' : GRAPH_FAILED.has(state) ? '[!]' : GRAPH_SUCCESS.has(state) ? '[+]' : '[.]'
  const tone = stuckState ? 'warn' : nodeRecord.in_frontier ? 'ok' : marker === '[!]' ? 'error' : marker === '[+]' ? 'muted' : 'primary'
  const text = `${safeText(nodeRecord.node_id)} · ${state}${stuckReason ? ` (${stuckReason})` : ''}${blockedBy.length ? ` · blocked_by ${blockedBy.map(safeText).join(' + ')}` : ''}`
  return { marker, tone, text }
}

function GraphSection({ graph }) {
  if (!graph || graph.status !== 'active') return null
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : []
  const shown = nodes.slice(0, GRAPH_NODE_LIMIT)
  const frontier = Array.isArray(graph.frontier) ? graph.frontier : []
  const edges = Array.isArray(graph.edges) ? graph.edges : []
  const edgeCount = Math.max(edges.length, Number(graph.edge_count) || 0)
  const hidden = Math.max(0, nodes.length - shown.length) + (Number(graph.hidden_nodes) || 0)
  const title = `${graph.scope === 'global' ? '[global] ' : ''}DAG · ${frontier.length} ready · ${edgeCount} edges`
  return [
    div('omh-pane-rule', [], 'graph-rule'),
    jsx(SectionHeader, { label: title, count: hidden ? `+${hidden} more` : '' }, 'graph-head'),
    div(
      'omh-pane-graph',
      shown.map((nodeRecord, index) => {
        const view = graphNodeView(nodeRecord)
        return div(`omh-pane-graph-node ${toneClass(view.tone)}`, [span('', view.marker), span('', view.text)], `${safeText(nodeRecord.node_id)}-${index}`)
      }),
      'graph-nodes'
    )
  ]
}

function HudPane() {
  const hud = useHud()
  const model = paneModel(hud)
  const hostModel = useValue(host.state.model)
  const usage = useValue(host.state.focusedUsage)
  const payload = model.payload
  const body = []
  if (model.loading) {
    body.push(jsx(Skeleton, { className: 'omh-pane-skeleton', style: { width: '70%' } }, 'skeleton-1'))
    body.push(jsx(Skeleton, { className: 'omh-pane-skeleton', style: { width: '45%' } }, 'skeleton-2'))
  } else if (payload) {
    body.push(jsx(PlanSection, { payload }, 'plan'))
    body.push(jsx(AgentsSection, { payload, receivedAt: hud.receivedAt }, 'agents'))
    body.push(jsx(GraphSection, { graph: payload.graph }, 'graph'))
  }
  return div('omh-pane', [
    jsx(Header, { model }),
    jsx(SessionLine, { sessionId: hud.sessionId, model: hostModel, usage }),
    payload ? jsx(Chips, { payload }) : null,
    div('omh-pane-rule', [], 'head-rule'),
    div('omh-pane-body', body),
    model.notice ? div('omh-pane-notice', model.notice) : null
  ])
}

function StatusItem() {
  const hud = useHud()
  const item = statusItem(hud)
  if (!item) return null
  const reveal = () => {
    if (typeof host.revealPane === 'function') host.revealPane(`${PLUGIN_ID}:${PANE_ID}`)
    else if (typeof host.notify === 'function') host.notify({ kind: 'info', message: item.detail })
  }
  return jsx(Tip, {
    label: item.detail,
    children: node('button', { type: 'button', className: 'omh-bar', onClick: reveal }, [
      span('omh-bar-glyph', '⚚'),
      ...item.segments.map(segment => span(`omh-bar-seg ${toneClass(segment.tone)}`, segment.text, segment.text))
    ])
  })
}

export default {
  id: PLUGIN_ID,
  name: 'oh-my-hermes',
  description: 'OMH plan, agent routes and status, read from the omh plugin backend inside the Hermes gateway.',
  defaultEnabled: false,
  register(ctx) {
    rest = ctx.rest
    const removeStyle = injectStyle()
    ctx.onDispose(() => {
      removeStyle()
      rest = null
    })
    ctx.registerMany([
      {
        id: 'status',
        area: STATUSBAR_AREAS.right,
        order: 130,
        render: () => jsx(StatusItem, {})
      },
      {
        id: PANE_ID,
        area: PANES_AREA,
        title: 'omh',
        data: { placement: 'right', width: PANE_WIDTH },
        render: () => jsx(HudPane, {})
      }
    ])
  }
}
