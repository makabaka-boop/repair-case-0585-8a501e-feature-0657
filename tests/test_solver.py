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


# --------------------------------------------------------------------------
# continuity objective: exhaustive oracle, locked case, references
# --------------------------------------------------------------------------


def enumerate_plans(n, max_len):
    """Every legal cover of [0, n) as ((start, end, source_id), ...).

    Enumerates all admissible cut positions and both sources per segment.
    """

    def rec(start, acc):
        for end in range(start + 1, min(start + max_len, n) + 1):
            for s in (SOURCE_A, SOURCE_B):
                acc.append((start, end, s))
                if end == n:
                    yield tuple(acc)
                else:
                    yield from rec(end, acc)
                acc.pop()

    yield from rec(0, [])


def _adjudication_keys(plan, costs, fees, objective):
    """Prefix adjudication tuples at every endpoint, newest first.

    Comparing two plans by this reversed list reproduces the recursive
    "ties defer to the prefix in the same order" rule for each objective.
    """
    pref = [[0], [0]]
    for a, b in costs:
        pref[0].append(pref[0][-1] + a)
        pref[1].append(pref[1][-1] + b)
    keys = []
    total = 0
    for idx, (a, b, s) in enumerate(plan, start=1):
        total += fees[s] + pref[s][b] - pref[s][a]
        if objective == "default":
            keys.append((total, idx, a, s))
        else:
            switches = sum(
                1 for t in range(1, idx) if plan[t][2] != plan[t - 1][2])
            keys.append((total, switches, idx, a, s))
    return keys[::-1]


def objective_oracle(n, costs, fee_a, fee_b, max_len, objective):
    """Brute force over every segmentation and source assignment."""
    fees = (fee_a, fee_b)
    best_keys = None
    best_plan = None
    for plan in enumerate_plans(n, max_len):
        keys = _adjudication_keys(plan, costs, fees, objective)
        if best_keys is None or keys < best_keys:
            best_keys, best_plan = keys, plan
    cost = best_keys[::-1][-1][0]
    return cost, [(a, b, "AB"[s]) for a, b, s in best_plan]


def continuity_reference_naive(n, costs, fee_a, fee_b, max_len):
    """Independent O(nL) continuity DP with full recursive prefix ranks."""
    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    EMPTY = (0, 0, 0, -1, -1)
    # state 2*j+p: best prefix ending in p covering [0,j)
    cost = [0] * (2 * (n + 1))
    switch = [0] * (2 * (n + 1))
    count = [0] * (2 * (n + 1))
    prev = [-1] * (2 * (n + 1))
    pprev = [-1] * (2 * (n + 1))
    reachable = [False] * (2 * (n + 1))

    for j in range(1, n + 1):
        for s in (SOURCE_A, SOURCE_B):
            best = None
            if j <= max_len:
                best = (pref[s][j] + fees[s], 0, 1, 0, EMPTY)
            for i in range(max(0, j - max_len), j):
                for p in (SOURCE_A, SOURCE_B):
                    k = 2 * i + p
                    if not reachable[k] and i != 0:
                        continue
                    if i == 0:
                        continue
                    rank = (cost[k], switch[k], count[k], prev[k], p)
                    cand = (
                        cost[k] + fees[s] + pref[s][j] - pref[s][i],
                        switch[k] + (1 if p != s else 0),
                        count[k] + 1,
                        i,
                        rank,
                    )
                    if best is None or cand < best:
                        best = cand
            c, w, m, i, rank = best
            out = 2 * j + s
            cost[out], switch[out], count[out] = c, w, m
            prev[out], pprev[out] = i, rank[-1]
            reachable[out] = True

    finals = [
        (cost[2 * n + s], switch[2 * n + s], count[2 * n + s],
         prev[2 * n + s], s)
        for s in (SOURCE_A, SOURCE_B)
    ]
    c, _, _, _, last = min(finals)
    segs = []
    j, p = n, last
    while j > 0:
        k = 2 * j + p
        segs.append((prev[k], j, "AB"[p]))
        p = pprev[k]
        j = prev[k]
    segs.reverse()
    return c, segs


def continuity_reference_heap(n, costs, fee_a, fee_b, max_len):
    """Independent O(n log n) continuity DP with lazy expiry heaps."""
    import heapq

    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    size = 2 * (n + 1)
    costs_arr = [0] * size
    sw_arr = [0] * size
    counts = [0] * size
    prev_i = [-1] * size
    prev_p = [-1] * size
    heaps = [[[] for _ in (SOURCE_A, SOURCE_B)] for _ in (SOURCE_A, SOURCE_B)]

    EMPTY = (0, 0, 0, -1, -1)
    for j in range(1, n + 1):
        low = j - max_len
        for s in (SOURCE_A, SOURCE_B):
            best = None
            if j <= max_len:
                best = (pref[s][j] + fees[s], 0, 1, 0, EMPTY)
            for p in (SOURCE_A, SOURCE_B):
                heap = heaps[p][s]
                while heap and heap[0][3] < low:
                    heapq.heappop(heap)
                if not heap:
                    continue
                _val, _sw, _cnt, i = heap[0]
                k = 2 * i + p
                cand = (
                    costs_arr[k] + fees[s] + pref[s][j] - pref[s][i],
                    sw_arr[k] + (1 if p != s else 0),
                    counts[k] + 1,
                    i,
                    (costs_arr[k], sw_arr[k], counts[k], prev_i[k], p),
                )
                if best is None or cand < best:
                    best = cand
            c, w, m, i, rank = best
            out = 2 * j + s
            costs_arr[out], sw_arr[out], counts[out] = c, w, m
            prev_i[out], prev_p[out] = i, rank[-1]
        for p in (SOURCE_A, SOURCE_B):
            out = 2 * j + p
            for s in (SOURCE_A, SOURCE_B):
                heapq.heappush(
                    heaps[p][s],
                    (costs_arr[out] - pref[s][j], sw_arr[out],
                     counts[out], j))

    finals = [
        (costs_arr[2 * n + s], sw_arr[2 * n + s], counts[2 * n + s],
         prev_i[2 * n + s], s)
        for s in (SOURCE_A, SOURCE_B)
    ]
    c, _, _, _, last = min(finals)
    segs = []
    j, p = n, last
    while j > 0:
        k = 2 * j + p
        segs.append((prev_i[k], j, "AB"[p]))
        p = prev_p[k]
        j = prev_i[k]
    segs.reverse()
    return c, segs


def _seg_tuples(solution):
    return [(s.start, s.end, s.source) for s in solution.segments]


@pytest.mark.parametrize("n", range(1, 10))
def test_continuity_matches_exhaustive_segmentation_oracle(n):
    # Enumerate every legal segmentation with every A/B assignment under
    # tie-heavy costs/fees; the solver must reproduce the full recursive
    # adjudication exactly.
    rng = random.Random(7000 + n)
    for trial in range(12):
        L = rng.randrange(1, n + 1)
        flavor = rng.random()
        if flavor < 0.4:
            costs = [(0, 0)] * n
        elif flavor < 0.8:
            costs = [(rng.choice((0, 1)), rng.choice((0, 1)))
                     for _ in range(n)]
        else:
            costs = [(rng.randrange(0, 4), rng.randrange(0, 4))
                     for _ in range(n)]
        fa = rng.choice((0, 0, 1, 4))
        fb = rng.choice((0, 0, 1, 4))
        for objective in ("default", "continuity"):
            sol = solve(n, costs, fa, fb, L, objective)
            ocost, osegs = objective_oracle(
                n, costs, fa, fb, L, objective)
            assert sol.cost == ocost
            assert _seg_tuples(sol) == osegs
            assert_plan_valid(sol.segments, n, L, costs, (fa, fb), sol.cost)


def test_locked_n3_L1_zero_costs_switch_pattern():
    n, L = 3, 1
    costs = [(0, 0), (1, 0), (0, 1)]
    default_plan = solve(n, costs, 0, 0, L)
    continuity_plan = solve(n, costs, 0, 0, L, "continuity")
    # Default (cost, segments, prev, source): A beats B at every position.
    assert default_plan.cost == 0
    assert _seg_tuples(default_plan) == [
        (0, 1, "A"), (1, 2, "B"), (2, 3, "A")]
    # Continuity: minimum cost 0; one switch beats two, and the switch
    # lands at the latest break with the final source equal to A: BBA.
    assert continuity_plan.cost == 0
    assert _seg_tuples(continuity_plan) == [
        (0, 1, "B"), (1, 2, "B"), (2, 3, "A")]


def test_continuity_against_naive_reference_medium():
    rng = random.Random(31337)
    for _ in range(40):
        n = rng.randrange(10, 70)
        L = rng.randrange(1, min(n, 14) + 1)
        costs = [(rng.randrange(0, 100), rng.randrange(0, 100))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 60), rng.randrange(0, 60)
        sol = solve(n, costs, fa, fb, L, "continuity")
        rc, rs = continuity_reference_naive(n, costs, fa, fb, L)
        assert sol.cost == rc
        assert _seg_tuples(sol) == rs
        assert_plan_valid(sol.segments, n, L, costs, (fa, fb), sol.cost)


def test_continuity_small_L_max_scale_matches_naive():
    n, L = 200_000, 3
    rng = random.Random(31338)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = 500_000, 500_000
    sol = solve(n, costs, fa, fb, L, "continuity")
    rc, rs = continuity_reference_naive(n, costs, fa, fb, L)
    assert sol.cost == rc
    assert _seg_tuples(sol) == rs
    assert_plan_valid(sol.segments, n, L, costs, (fa, fb), sol.cost)


def test_continuity_max_scale_performance_and_heap_reference():
    n, L = 200_000, 4096
    rng = random.Random(31339)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001)

    t0 = time.perf_counter()
    sol = solve(n, costs, fa, fb, L, "continuity")
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0  # O(n), same budget as the default max-scale case
    assert_plan_valid(sol.segments, n, L, costs, (fa, fb), sol.cost)

    rc, rs = continuity_reference_heap(n, costs, fa, fb, L)
    assert sol.cost == rc
    assert _seg_tuples(sol) == rs

    # Continuity minimizes cost first: its minimum total cost cannot be
    # below the default objective's minimum cost (same feasible set).
    default_sol = solve(n, costs, fa, fb, L, "default")
    assert default_sol.cost == sol.cost


# --------------------------------------------------------------------------
# objective on the API: legality, 422 shape, default compatibility
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_objective", [
    "fastest", "DEFAULT", "Default", " continuity", "continuity ",
    "", "min-cost", 3, True, None,
])
def test_api_illegal_objective_422(bad_objective):
    payload = _payload(objective=bad_objective)
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    # Only the standard validation payload; never a partial plan.
    assert set(body) == {"detail"}
    assert "cost" not in body and "segments" not in body
    assert body["detail"][0]["loc"][-1] == "objective"


def test_api_explicit_default_matches_omitted():
    omitted = client.post("/solve", json=_payload())
    explicit = client.post("/solve", json=_payload(objective="default"))
    assert omitted.status_code == explicit.status_code == 200
    assert omitted.json() == explicit.json()


def test_api_default_compatibility_matches_oracle_random():
    rng = random.Random(8080)
    for _ in range(12):
        n = rng.randrange(1, 40)
        L = rng.randrange(1, n + 1)
        fa, fb = rng.randrange(0, 50), rng.randrange(0, 50)
        costs = [(rng.randrange(0, 100), rng.randrange(0, 100))
                 for _ in range(n)]
        payload = {
            "n": n,
            "L": L,
            "fee_a": fa,
            "fee_b": fb,
            "costs": [{"a": a, "b": b} for a, b in costs],
        }
        omitted = client.post("/solve", json=payload)
        explicit = client.post("/solve", json={**payload, "objective": "default"})
        continuity = client.post("/solve", json={**payload, "objective": "continuity"})
        assert omitted.status_code == explicit.status_code == 200
        assert omitted.json() == explicit.json()
        ocost, osegs = oracle(n, costs, fa, fb, L)
        assert omitted.json()["cost"] == ocost
        assert [(s["start"], s["end"], s["source"])
                for s in omitted.json()["segments"]] == osegs
        assert continuity.status_code == 200
        assert set(continuity.json()) == {"cost", "segments"}
        assert continuity.json()["cost"] == ocost


def test_api_continuity_locked_case():
    n, L = 3, 1
    costs = [(0, 0), (1, 0), (0, 1)]
    payload = {
        "n": n, "L": L, "fee_a": 0, "fee_b": 0,
        "costs": [{"a": a, "b": b} for a, b in costs],
    }
    default_resp = client.post("/solve", json=payload)
    continuity_resp = client.post("/solve", json={**payload, "objective": "continuity"})
    assert default_resp.json()["segments"] == [
        {"start": 0, "end": 1, "source": "A"},
        {"start": 1, "end": 2, "source": "B"},
        {"start": 2, "end": 3, "source": "A"},
    ]
    assert continuity_resp.json()["segments"] == [
        {"start": 0, "end": 1, "source": "B"},
        {"start": 1, "end": 2, "source": "B"},
        {"start": 2, "end": 3, "source": "A"},
    ]
