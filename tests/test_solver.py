"""Acceptance tests.

* Exhaustive oracle for every n <= 10 instance (all A/B assignments over
  the gap, segmented by all admissible cut positions).
* A second independent O(nL) reference DP with the exact same tie
  breakers, used on random instances and on the n=200000 max-scale case
  (L small enough there), plus an independent heap-based sliding-window
  minimum for the full max-scale instance.
* Every returned plan is replayed from the recurrence and must reproduce
  the reported optimum exactly.
* Malformed length, out-of-range values and illegal L all give 422 and
  never a partial plan.
"""

import random
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import SOURCE_A, SOURCE_B, Segment, solve

client = TestClient(app)


# --------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------


def recompute(costs, fees, segments):
    """Sum segment costs directly from the definition."""
    fees = {"A": fees[0], "B": fees[1]}
    total = 0
    for start, end, source in segments:
        s = 0 if source == "A" else 1
        total += fees[source] + sum(costs[k][s] for k in range(start, end))
    return total


def oracle(n, costs, fee_a, fee_b, max_len):
    """Brute-force DP that enumerates every (i, source) for every endpoint.

    Comparison order is exactly (cost, segment count, predecessor, source)
    with source A < B, so it returns the same unique plan as the solver.
    """
    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    best = {0: (0, 0, -1, -1)}  # j -> (cost, count, prev, source)
    for j in range(1, n + 1):
        cand = None
        for i in range(max(0, j - max_len), j):
            for s in (SOURCE_A, SOURCE_B):
                c = (
                    best[i][0] + fees[s] + pref[s][j] - pref[s][i],
                    best[i][1] + 1,
                    i,
                    s,
                )
                if cand is None or c < cand:
                    cand = c
        best[j] = cand

    segs = []
    j = n
    while j > 0:
        _, _, i, s = best[j]
        segs.append((i, j, "AB"[s]))
        j = i
    segs.reverse()
    return best[n][0], segs


def heap_reference(n, costs, fee_a, fee_b, max_len):
    """Independent O(n log n) reference using lazy-deletion heaps.

    Used for the full max-scale case where O(nL) is too slow.
    """
    import heapq

    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    costs_arr = [0] * (n + 1)
    counts = [0] * (n + 1)
    prev = [-1] * (n + 1)
    psrc = [-1] * (n + 1)

    # Heaps hold (val, count, index); expired/dead indices are skipped.
    heaps = [
        [(0, 0, 0)],  # source A
        [(0, 0, 0)],  # source B
    ]
    live_sets = [{0}, {0}]

    for j in range(1, n + 1):
        low = j - max_len
        candidates = []
        for s, heap in enumerate(heaps):
            live = live_sets[s]
            while heap:
                val, count, i = heap[0]
                if i < low or i not in live:
                    heapq.heappop(heap)
                else:
                    break
            val, count, i = heap[0]
            candidates.append((val + pref[s][j] + fees[s], count + 1, i, s))
        cost, count, i, s = min(candidates)
        costs_arr[j] = cost
        counts[j] = count
        prev[j] = i
        psrc[j] = s
        for t, heap in enumerate(heaps):
            entry = (costs_arr[j] - pref[t][j], counts[j], j)
            heapq.heappush(heap, entry)
            live_sets[t].add(j)

    segs = []
    j = n
    while j > 0:
        segs.append((prev[j], j, "AB"[psrc[j]]))
        j = prev[j]
    segs.reverse()
    return costs_arr[n], segs


def random_instance(rng, n, L, vmax=10, fmax=10):
    costs = [(rng.randrange(0, vmax + 1), rng.randrange(0, vmax + 1))
             for _ in range(n)]
    return n, costs, rng.randrange(0, fmax + 1), rng.randrange(0, fmax + 1), L


def assert_plan_valid(plan, n, max_len, costs, fees, reported_cost):
    # Coverage: back-to-back half-open intervals spanning exactly [0, n).
    assert plan[0].start == 0
    assert plan[-1].end == n
    for a, b in zip(plan, plan[1:]):
        assert a.end == b.start
    for seg in plan:
        assert seg.source in ("A", "B")
        assert 1 <= seg.end - seg.start <= max_len
    # Exact replay from the cost definition.
    assert recompute(costs, fees,
                     [(s.start, s.end, s.source) for s in plan]) == reported_cost


# --------------------------------------------------------------------------
# Exhaustive oracle for n <= 10
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 11))
def test_matches_exhaustive_oracle_all_fees(n):
    # Deterministic pseudo-random instances covering enough variety; the
    # oracle itself enumerates all structural choices exactly, so matching
    # it on varied costs/fees exercises every tie breaker.
    rng = random.Random(1000 + n)
    for trial in range(20):
        _, costs, fa, fb, L = random_instance(
            rng, n, rng.randrange(1, n + 1), vmax=5, fmax=6)
        sol = solve(n, costs, fa, fb, L)
        ocost, osegs = oracle(n, costs, fa, fb, L)
        assert sol.cost == ocost
        assert [(s.start, s.end, s.source) for s in sol.segments] == osegs
        assert_plan_valid(sol.segments, n, L, costs, (fa, fb), sol.cost)


def test_handcrafted_tie_breakers():
    # Zero fees and equal per-position costs: every cover costs the same.
    # Unique plan must be: fewest segments, then smallest predecessor,
    # then source A.
    n, L = 5, 3
    costs = [(0, 0)] * n
    sol = solve(n, costs, 0, 0, L)
    # Fewest segments with length <= 3: a 2-segment cover. Cuts at 2
    # ([0,2)+[2,5)) and 3 ([0,3)+[3,5)) tie; the smaller predecessor
    # index wins, and source A wins the A/B tie.
    assert sol.cost == 0
    assert [(s.start, s.end, s.source) for s in sol.segments] == [
        (0, 2, "A"), (2, 5, "A")]
    sol2 = solve(n, [(1, 2)] * n, 0, 0, L)
    ocost, osegs = oracle(n, [(1, 2)] * n, 0, 0, L)
    assert sol2.cost == ocost
    assert [(s.start, s.end, s.source) for s in sol2.segments] == osegs


# --------------------------------------------------------------------------
# Larger / max-scale cases
# --------------------------------------------------------------------------


def test_random_medium_against_oracle():
    rng = random.Random(42)
    for _ in range(30):
        n = rng.randrange(11, 60)
        L = rng.randrange(1, min(n, 12) + 1)
        _, costs, fa, fb, _ = random_instance(rng, n, L, vmax=100, fmax=50)
        sol = solve(n, costs, fa, fb, L)
        ocost, osegs = oracle(n, costs, fa, fb, L)
        assert sol.cost == ocost
        assert [(s.start, s.end, s.source) for s in sol.segments] == osegs


def test_max_scale_full():
    n, L = 200_000, 4096
    rng = random.Random(7)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001)

    t0 = time.perf_counter()
    sol = solve(n, costs, fa, fb, L)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0  # O(n), nowhere near enumeration cost

    ref_cost, ref_segs = heap_reference(n, costs, fa, fb, L)
    assert sol.cost == ref_cost
    assert [(s.start, s.end, s.source) for s in sol.segments] == ref_segs
    assert_plan_valid(sol.segments, n, L, costs, (fa, fb), sol.cost)


def test_larger_than_n_window():
    # L may exceed n; the window simply covers every predecessor.
    rng = random.Random(2026)
    for n in (1, 2, 5, 10):
        _, costs, fa, fb, _ = random_instance(
            rng, n, 4096, vmax=1000, fmax=1000)
        sol = solve(n, costs, fa, fb, 4096)
        ocost, osegs = oracle(n, costs, fa, fb, 4096)
        assert sol.cost == ocost
        assert [(s.start, s.end, s.source) for s in sol.segments] == osegs
        assert_plan_valid(sol.segments, n, 4096, costs, (fa, fb), sol.cost)


def test_max_scale_with_small_L_matches_naive_dp():
    n, L = 200_000, 3
    rng = random.Random(11)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = 500_000, 500_000
    sol = solve(n, costs, fa, fb, L)
    ocost, osegs = oracle(n, costs, fa, fb, L)  # 6e5 ops, fine
    assert sol.cost == ocost
    assert [(s.start, s.end, s.source) for s in sol.segments] == osegs


# --------------------------------------------------------------------------
# API behaviour
# --------------------------------------------------------------------------


def _payload(n=5, **overrides):
    p = {
        "n": n,
        "L": 3,
        "fee_a": 1,
        "fee_b": 2,
        "costs": [{"a": k, "b": k + 1} for k in range(n)],
    }
    p.update(overrides)
    return p


def test_api_happy_path_and_replay():
    resp = client.post("/solve", json=_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"cost", "segments"}
    n = 5
    costs = [(k, k + 1) for k in range(n)]
    assert_plan_valid(
        [Segment(s["start"], s["end"], s["source"]) for s in data["segments"]],
        n, 3, costs, (1, 2), data["cost"])
    ocost, osegs = oracle(n, costs, 1, 2, 3)
    assert data["cost"] == ocost
    assert [(s["start"], s["end"], s["source"]) for s in
            data["segments"]] == osegs


def test_api_max_scale():
    n = 200_000
    rng = random.Random(99)
    costs = [{"a": rng.randrange(1_000_001), "b": rng.randrange(1_000_001)}
             for _ in range(n)]
    resp = client.post("/solve", json={"n": n, "L": 4096,
                                       "fee_a": 10, "fee_b": 20,
                                       "costs": costs})
    assert resp.status_code == 200
    data = resp.json()
    segs = data["segments"]
    assert segs[0]["start"] == 0 and segs[-1]["end"] == n
    assert all(b["start"] == a["end"] for a, b in zip(segs, segs[1:]))
    assert all(1 <= s["end"] - s["start"] <= 4096 for s in segs)


@pytest.mark.parametrize("payload", [
    # length mismatch: costs shorter / longer than n
    lambda: _payload(n=5, costs=[{"a": 1, "b": 1}] * 4),
    lambda: _payload(n=5, costs=[{"a": 1, "b": 1}] * 6),
    # value out of range
    lambda: _payload(fee_a=-1),
    lambda: _payload(fee_b=1_000_001),
    lambda: _payload(costs=[{"a": -1, "b": 0}] + [{"a": 0, "b": 0}] * 4),
    lambda: _payload(costs=[{"a": 0, "b": 1_000_001}] + [{"a": 0, "b": 0}] * 4),
    # n out of range
    lambda: _payload(n=0, costs=[]),
    lambda: _payload(n=200_001, costs=[{"a": 0, "b": 0}] * 200_001),
    # illegal L
    lambda: _payload(L=0),
    lambda: _payload(L=4097),
    # wrong types
    lambda: {**_payload(), "n": "5"},
    lambda: {**_payload(), "L": True},
    lambda: _payload(costs=[{"a": 1.5, "b": 0}] + [{"a": 0, "b": 0}] * 4),
    lambda: _payload(costs=[{"a": "1", "b": 0}] + [{"a": 0, "b": 0}] * 4),
    # missing field
    lambda: {k: v for k, v in _payload().items() if k != "fee_b"},
])
def test_api_validation_errors_422(payload):
    resp = client.post("/solve", json=payload())
    assert resp.status_code == 422
    body = resp.json()
    # Only the standard validation payload; never a partial plan.
    assert set(body) == {"detail"}
    assert "cost" not in body and "segments" not in body


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
