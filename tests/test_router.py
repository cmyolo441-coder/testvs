"""Routing: ordering strategies, failover, cooldowns and hedging."""

from __future__ import annotations

import tempfile
import time
import unittest

from omh.config import ModelProfile
from omh.errors import AuthError, ProviderError, StreamInterrupted
from omh.providers.base import Chunk, Usage
from omh.router import ModelRouter

try:  # discovery from the repo root imports this as tests.test_router
    from .support import ScriptedProvider, make_config, responder
except ImportError:  # pragma: no cover - direct module loading
    from support import ScriptedProvider, make_config, responder


class Clock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def build_router(
    script_a,
    script_b=None,
    *,
    strategy: str = "priority",
    hedge: float = 0.0,
    weights=(2.0, 1.0),
    clock=None,
):
    tmp = tempfile.mkdtemp()
    config = make_config(
        tmp,
        models=[
            ModelProfile("model-a", "a", label="a", weight=weights[0]),
            ModelProfile("model-b", "b", label="b", weight=weights[1]),
        ],
    )
    config.router.strategy = strategy
    config.router.hedge_delay_s = hedge
    config.router.max_attempts = 2
    provider_a = ScriptedProvider("a", script_a)
    provider_b = ScriptedProvider("b", script_b or [responder("b", model="model-b")])
    router = ModelRouter(
        config, {"a": provider_a, "b": provider_b}, clock=clock or time.monotonic
    )
    return router, provider_a, provider_b


def collect(router: ModelRouter) -> str:
    return "".join(chunk.text for chunk in router.stream([], None))


class OrderingTest(unittest.TestCase):
    def test_priority_follows_config_order(self) -> None:
        router, _, _ = build_router([responder("a")])
        self.assertEqual([state.label for state in router.order()], ["a", "b"])

    def test_balanced_prefers_weight_when_unmeasured(self) -> None:
        router, _, _ = build_router([responder("a")], strategy="balanced")
        self.assertEqual(router.order()[0].label, "a")

    def test_balanced_uses_ewma_latency(self) -> None:
        router, _, _ = build_router([responder("a")], strategy="balanced")
        router.states[0].latency = 2.0
        router.states[1].latency = 0.2
        self.assertEqual(router.order()[0].label, "b")

    def test_round_robin_is_weighted(self) -> None:
        router, _, _ = build_router([responder("a")], strategy="round_robin")
        picks = [router.order()[0].label for _ in range(6)]
        self.assertIn("a", picks)
        self.assertIn("b", picks)
        self.assertGreater(picks.count("a"), picks.count("b"))

    def test_provider_filter_limits_pool(self) -> None:
        router, _, _ = build_router([responder("a")])
        router.config.provider = "b"
        router.states = [state for state in router.states if state.profile.provider == "b"]
        self.assertEqual([state.label for state in router.order()], ["b"])


class FailoverTest(unittest.TestCase):
    def test_failover_to_second_provider(self) -> None:
        clock = Clock()
        router, provider_a, provider_b = build_router(
            [ProviderError("boom", status=503)], clock=clock
        )
        self.assertEqual(collect(router), "b")
        self.assertEqual(provider_a.calls, 1)
        self.assertEqual(provider_b.calls, 1)
        self.assertEqual(router.states[0].failures, 1)
        self.assertEqual(router.states[1].successes, 1)
        self.assertTrue(router.failover_notes)
        self.assertGreater(router.states[0].cooldown_until, clock.now)

    def test_cooldown_skips_failed_lane(self) -> None:
        clock = Clock()
        router, _, _ = build_router([ProviderError("boom", status=500)], clock=clock)
        collect(router)
        self.assertEqual(router.order()[0].label, "b")
        clock.advance(router.config.router.cooldown_s + 1)
        self.assertIn("a", [state.label for state in router.healthy()])

    def test_auth_error_uses_long_cooldown(self) -> None:
        clock = Clock()
        router, _, _ = build_router([AuthError("bad key")], clock=clock)
        collect(router)
        self.assertGreaterEqual(router.states[0].cooldown_until - clock.now, 300)

    def test_all_lanes_failing_reports_both(self) -> None:
        router, _, _ = build_router([ProviderError("one")], [ProviderError("two")])
        with self.assertRaises(ProviderError) as caught:
            collect(router)
        self.assertIn("every model", caught.exception.message)
        self.assertIn("a", caught.exception.hint)
        self.assertIn("b", caught.exception.hint)

    def test_partial_stream_interrupts_without_retrying(self) -> None:
        def partial(**_) -> None:
            yield Chunk(text="half", model="model-a")
            raise ProviderError("cut")

        router, _, provider_b = build_router([partial])
        chunks = router.stream([], None)
        first = next(chunks)
        self.assertEqual(first.text, "half")
        with self.assertRaises(StreamInterrupted):
            next(chunks)
        self.assertEqual(provider_b.calls, 0)

    def test_no_models_available(self) -> None:
        router, _, _ = build_router([responder("a")])
        router.states = []
        with self.assertRaises(ProviderError) as caught:
            collect(router)
        self.assertIn("no model is available", caught.exception.message)

    def test_usage_is_counted(self) -> None:
        def with_usage(**_) -> None:
            yield Chunk(text="ok", model="model-a")
            yield Chunk(finish="stop", usage=Usage(10, 5), model="model-a")

        router, _, _ = build_router([with_usage])
        self.assertEqual(collect(router), "ok")
        self.assertEqual(router.states[0].tokens, 15)
        self.assertEqual(router.profile_for("model-a").label, "a")


class HedgingTest(unittest.TestCase):
    def test_fast_lane_wins_the_race(self) -> None:
        router, provider_a, provider_b = build_router(
            [responder("slow", model="model-a", delay=0.3)],
            [responder("fast", model="model-b")],
            hedge=0.05,
        )
        started = time.monotonic()
        output = collect(router)
        elapsed = time.monotonic() - started
        self.assertEqual(output, "fast")
        self.assertLess(elapsed, 0.25)
        self.assertEqual(router.states[1].successes, 1)
        self.assertEqual(provider_a.calls, 1)
        self.assertEqual(provider_b.calls, 1)

    def test_hedged_failure_falls_back(self) -> None:
        router, _, _ = build_router(
            [ProviderError("down")], [responder("backup")], hedge=0.05
        )
        self.assertEqual(collect(router), "backup")

    def test_hedged_all_fail(self) -> None:
        router, _, _ = build_router(
            [ProviderError("one")], [ProviderError("two")], hedge=0.05
        )
        with self.assertRaises(ProviderError):
            collect(router)

    def test_hedging_does_not_duplicate_text(self) -> None:
        def two_chunks(**_) -> None:
            yield Chunk(text="fast-", model="model-b")
            yield Chunk(text="lane", model="model-b")
            yield Chunk(finish="stop", model="model-b")

        router, _, _ = build_router(
            [responder("slow", model="model-a", delay=0.3)],
            [two_chunks],
            hedge=0.05,
        )
        self.assertEqual(collect(router), "fast-lane")


if __name__ == "__main__":
    unittest.main()

