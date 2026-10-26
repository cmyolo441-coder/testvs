"""Model pool: weighted load balancing, failover, cooldowns and hedging.

Three strategies:

``priority``     always the first healthy profile (predictable, cheapest).
``round_robin``  smooth weighted round robin (fair, no latency input).
``balanced``     weight / latency-EWMA, so fast lanes get more traffic while
                 every healthy lane keeps being sampled.

Every attempt is timed. A failure puts that lane in an exponential cooldown;
the next candidate is tried before any text has been shown, so a dead provider
is invisible to the user. With ``hedge_delay_s > 0`` a second lane is started
when the first has not produced a token in time, and the fastest wins.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from .config import Config, ModelProfile
from .errors import AuthError, ProviderError, StreamInterrupted
from .providers.base import ChatMessage, Chunk, Provider, Usage


@dataclass
class ModelState:
    profile: ModelProfile
    provider: Provider
    requests: int = 0
    successes: int = 0
    failures: int = 0
    tokens: int = 0
    latency: float = 0.0  # EWMA time-to-first-token, seconds
    last_latency: float = 0.0
    cooldown_until: float = 0.0
    last_error: str = ""
    rr_current: float = 0.0

    @property
    def model_id(self) -> str:
        return self.profile.id

    @property
    def label(self) -> str:
        return self.profile.name

    def cooling(self, now: float) -> bool:
        return now < self.cooldown_until


class ModelRouter:
    def __init__(
        self,
        config: Config,
        providers: Dict[str, Provider],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.clock = clock
        self.states: List[ModelState] = []
        for profile in config.enabled_profiles():
            provider = providers.get(profile.provider)
            if provider is not None:
                self.states.append(ModelState(profile=profile, provider=provider))
        self.last_profile: Optional[ModelProfile] = None
        self.failover_notes: List[str] = []

    # -- introspection ---------------------------------------------------
    def available(self) -> bool:
        return bool(self.states)

    def healthy(self) -> List[ModelState]:
        now = self.clock()
        return [state for state in self.states if not state.cooling(now)]

    def _candidates(self) -> List[ModelState]:
        healthy = self.healthy()
        if healthy:
            return healthy
        # Everything is cooling down: use it anyway, with the real error.
        return sorted(self.states, key=lambda state: state.cooldown_until)

    def order(self) -> List[ModelState]:
        candidates = self._candidates()
        strategy = self.config.router.strategy
        if strategy == "priority":
            return list(candidates)
        if strategy == "round_robin":
            return self._round_robin(candidates)
        return sorted(candidates, key=self._score, reverse=True)

    def _round_robin(self, candidates: List[ModelState]) -> List[ModelState]:
        """Smooth weighted round robin, one pick per call."""
        total = sum(max(0.0, state.profile.weight) for state in candidates) or 1.0
        for state in candidates:
            state.rr_current += max(0.01, state.profile.weight)
        chosen = max(candidates, key=lambda state: state.rr_current)
        chosen.rr_current -= total
        others = [state for state in candidates if state is not chosen]
        return [chosen, *others]

    def _score(self, state: ModelState) -> float:
        weight = max(0.01, state.profile.weight)
        if state.latency <= 0:
            return weight * 4.0  # explore an unmeasured lane first
        return weight * 2.0 / (state.latency + 0.15)

    def profile_for(self, model_id: str) -> Optional[ModelProfile]:
        for state in self.states:
            if state.model_id == model_id:
                return state.profile
        return None

    def stats(self) -> List[Dict[str, Any]]:
        now = self.clock()
        rows = []
        for state in self.states:
            rows.append(
                {
                    "provider": state.profile.provider,
                    "model": state.label,
                    "weight": state.profile.weight,
                    "requests": state.requests,
                    "failures": state.failures,
                    "tokens": state.tokens,
                    "latency_ms": int(state.latency * 1000),
                    "health": "cooling" if state.cooling(now) else "ready",
                    "last_error": state.last_error,
                }
            )
        return rows


    # -- bookkeeping -----------------------------------------------------
    def _track_usage(self, state: ModelState, usage: Optional[Usage]) -> None:
        if usage is not None:
            state.tokens += usage.total

    def _record_success(self, state: ModelState, ttft: float) -> None:
        state.requests += 1
        state.successes += 1
        state.last_latency = max(0.0, ttft)
        state.latency = (
            state.last_latency
            if state.latency <= 0
            else 0.65 * state.latency + 0.35 * state.last_latency
        )
        state.cooldown_until = 0.0

    def _record_failure(self, state: ModelState, error: BaseException) -> None:
        state.requests += 1
        state.failures += 1
        message = getattr(error, "message", str(error)) or "unknown error"
        state.last_error = message
        retry_after = getattr(error, "retry_after", None)
        if isinstance(error, AuthError):
            cooldown = 300.0
        elif retry_after:
            cooldown = max(1.0, float(retry_after))
        else:
            cooldown = min(
                300.0,
                max(1.0, self.config.router.cooldown_s)
                * (2 ** min(4, max(0, state.failures - 1))),
            )
        state.cooldown_until = self.clock() + cooldown
        self.failover_notes.append(f"{state.label}: {message}")

    # -- streaming -------------------------------------------------------
    def stream(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Iterator[Chunk]:
        if not self.states:
            hints = self.config.missing_credentials()
            raise ProviderError(
                "no model is available",
                hint="; ".join(hints[:3])
                or "run `omh --offline` for a demo, or set an API key",
                retryable=False,
            )
        candidates = self.order()[: max(1, self.config.router.max_attempts)]
        hedge = self.config.router.hedge_delay_s
        if hedge > 0 and len(candidates) > 1:
            yield from self._hedged(candidates, messages, tools, temperature, max_tokens)
        else:
            yield from self._sequential(
                candidates, messages, tools, temperature, max_tokens
            )

    def _sequential(
        self,
        candidates: List[ModelState],
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[Chunk]:
        errors: List[str] = []
        emitted = False
        for state in candidates:
            started = self.clock()
            ttft: Optional[float] = None
            try:
                for chunk in state.provider.stream(
                    model=state.model_id,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if ttft is None:
                        ttft = self.clock() - started
                    self._track_usage(state, chunk.usage)
                    self.last_profile = state.profile
                    emitted = True
                    yield chunk
                self._record_success(state, ttft if ttft is not None else self.clock() - started)
                return
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except BaseException as exc:  # provider errors and bugs alike
                error = self._as_provider_error(exc, state)
                self._record_failure(state, error)
                errors.append(f"{state.label}: {error.message}")
                if emitted:
                    raise StreamInterrupted(
                        error.message,
                        hint="stream failed after partial output; retry the turn",
                    ) from exc

        raise ProviderError(
            "every model in the pool failed",
            hint=" | ".join(errors[:3]) or "check `/balance` and your API keys",
        )

    @staticmethod
    def _as_provider_error(exc: BaseException, state: ModelState) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        return ProviderError(
            f"{state.label}: {type(exc).__name__}: {exc}", retryable=True
        )


    def _worker(
        self,
        index: int,
        state: ModelState,
        cancel: threading.Event,
        events: "queue.Queue",
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> None:
        try:
            for chunk in state.provider.stream(
                model=state.model_id,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if cancel.is_set():
                    return
                events.put((index, "chunk", chunk))
            if not cancel.is_set():
                events.put((index, "end", None))
        except (KeyboardInterrupt, SystemExit, GeneratorExit):  # pragma: no cover
            raise
        except BaseException as exc:
            events.put((index, "error", exc))

    def _hedged(
        self,
        candidates: List[ModelState],
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[Chunk]:
        events: "queue.Queue" = queue.Queue()
        attempts: List[tuple] = []
        starts: Dict[int, float] = {}
        limit = min(len(candidates), max(1, self.config.router.max_attempts))
        hedge = max(0.05, self.config.router.hedge_delay_s)
        errors: List[str] = []

        def spawn(state: ModelState) -> int:
            index = len(attempts)
            cancel = threading.Event()
            attempts.append((state, cancel))
            starts[index] = self.clock()
            threading.Thread(
                target=self._worker,
                args=(index, state, cancel, events, messages, tools, temperature, max_tokens),
                daemon=True,
                name=f"omh-hedge-{index}",
            ).start()
            return index

        try:
            spawn(candidates[0])
            launched_at = self.clock()
            finished: set = set()
            winner: Optional[int] = None
            first_chunk: Optional[Chunk] = None

            while winner is None:
                if len(attempts) < limit:
                    remaining = hedge - (self.clock() - launched_at)
                    if remaining <= 0:
                        spawn(candidates[len(attempts)])
                        launched_at = self.clock()
                        continue
                    try:
                        item = events.get(timeout=remaining)
                    except queue.Empty:
                        continue
                else:
                    item = events.get()

                index, kind, payload = item
                if kind == "chunk":
                    if payload.text or payload.tool_calls or payload.finish:
                        winner = index
                        first_chunk = payload
                        for other, cancel in attempts:
                            if other != index:
                                cancel.set()
                        break
                    continue
                finished.add(index)
                if kind == "error":
                    state = attempts[index][0]
                    error = self._as_provider_error(payload, state)
                    self._record_failure(state, error)
                    errors.append(f"{state.label}: {error.message}")
                    if len(attempts) < limit:
                        spawn(candidates[len(attempts)])
                        launched_at = self.clock()
                        continue
                if len(finished) >= len(attempts) and len(attempts) >= limit:
                    raise ProviderError(
                        "every model in the pool failed",
                        hint=" | ".join(errors[:3]) or "check `/balance` and your API keys",
                    )

            state = attempts[winner][0]
            ttft = max(0.0, self.clock() - starts[winner])
            emitted = False
            if first_chunk is not None:
                self._track_usage(state, first_chunk.usage)
                self.last_profile = state.profile
                if first_chunk.text or first_chunk.tool_calls:
                    emitted = True
                yield first_chunk

            while True:
                index, kind, payload = events.get()
                if index != winner:
                    continue
                if kind == "chunk":
                    self._track_usage(state, payload.usage)
                    if payload.text or payload.tool_calls:
                        emitted = True
                    yield payload
                elif kind == "end":
                    self._record_success(state, ttft)
                    return
                else:
                    error = self._as_provider_error(payload, state)
                    self._record_failure(state, error)
                    if emitted:
                        raise StreamInterrupted(
                            error.message,
                            hint="hedged stream failed after partial output; retry",
                        ) from payload
                    raise error
        finally:
            for _, cancel in attempts:
                cancel.set()


