"""Sliding-window dynamic programs for satellite telemetry gap repair.

A gap of ``n`` positions ``[0, n)`` is covered by back-to-back half-open
segments ``[start, end)``. Every segment uses source A or B. Choosing source
``s`` for ``[i, j)`` costs

    fee[s] + sum(cost[k][s] for k in range(i, j))

so the activation fee is paid once per segment, including when adjacent
segments use the same source.

The default objective compares candidates lexicographically by
``(cost, segments, predecessor, source)``.

The ``continuity`` objective first keeps the minimum total cost. Among those
plans it minimizes source changes (the first segment is not a change), then
segment count, final-segment start, and final source. At each recurrence, a
tie in these values uses the selected prefix's own continuity ordering. It is
computed with separate A/B prefix states; the no-source sentinel is only used
for a first segment and is never inserted into an A/B state window.

Both objectives use O(n) time and O(n) space.
"""

from collections import deque
from dataclasses import dataclass

# Source ids double as tie-breakers: A < B.
SOURCE_A = 0
SOURCE_B = 1
SOURCE_NAMES = ("A", "B")

OBJECTIVE_DEFAULT = "default"
OBJECTIVE_CONTINUITY = "continuity"

_NO_SOURCE = -1


@dataclass(frozen=True)
class Segment:
    """A half-open segment [start, end) served by one source."""

    start: int
    end: int
    source: str

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "source": self.source}


@dataclass(frozen=True)
class Solution:
    cost: int
    segments: tuple[Segment, ...]


def solve(n: int, costs: list[tuple[int, int]], fee_a: int, fee_b: int,
          max_len: int, objective: str = OBJECTIVE_DEFAULT) -> Solution:
    """Compute the optimal cover of [0, n).

    ``costs[k]`` is the pair of per-position costs ``(cost of A, cost of
    B)`` at position ``k``. Inputs are assumed already validated by the API
    layer; the algorithm itself trusts the bounds.
    """
    if objective == OBJECTIVE_CONTINUITY:
        return _solve_continuity(n, costs, fee_a, fee_b, max_len)
    return _solve_default(n, costs, fee_a, fee_b, max_len)


def _prefix_sums(
    n: int, costs: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    prefix_a = [0] * (n + 1)
    prefix_b = [0] * (n + 1)
    for k in range(n):
        a, b = costs[k]
        prefix_a[k + 1] = prefix_a[k] + a
        prefix_b[k + 1] = prefix_b[k] + b
    return prefix_a, prefix_b


def _solve_default(n: int, costs: list[tuple[int, int]], fee_a: int,
                   fee_b: int, max_len: int) -> Solution:
    fees = (fee_a, fee_b)
    prefixes = _prefix_sums(n, costs)

    # DP tables.
    best_cost = [0] * (n + 1)
    best_seg_count = [0] * (n + 1)
    prev_index = [-1] * (n + 1)
    prev_source = [-1] * (n + 1)

    # Each entry in window s is an index i, keyed by
    # (best_cost[i] - prefix_s[i], best_seg_count[i], i).
    windows: tuple[deque, deque] = (deque([0]), deque([0]))

    for j in range(1, n + 1):
        low = j - max_len
        for window in windows:
            while window and window[0] < low:
                window.popleft()

        best_candidate = None  # (cost, segments, prev, source)
        for s, window in enumerate(windows):
            i = window[0]
            candidate = (
                best_cost[i] - prefixes[s][i] + prefixes[s][j] + fees[s],
                best_seg_count[i] + 1,
                i,
                s,
            )
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        cost, seg_count, i, s = best_candidate
        best_cost[j] = cost
        best_seg_count[j] = seg_count
        prev_index[j] = i
        prev_source[j] = s

        for s, window in enumerate(windows):
            key = (best_cost[j] - prefixes[s][j], best_seg_count[j])
            while window:
                tail = window[-1]
                tail_key = (
                    best_cost[tail] - prefixes[s][tail],
                    best_seg_count[tail],
                )
                # Equal keys retain the smaller/older index; it has the
                # predecessor-index tie breaker and expires first.
                if tail_key <= key:
                    break
                window.pop()
            window.append(j)

    return _reconstruct(n, prev_index, prev_source, best_cost[n])


def _solve_continuity(n: int, costs: list[tuple[int, int]], fee_a: int,
                      fee_b: int, max_len: int) -> Solution:
    fees = (fee_a, fee_b)
    prefixes = _prefix_sums(n, costs)

    # State p is the source of the last segment (0=A, 1=B). Arrays are
    # indexed [p][j]. The empty prefix is not state A or B.
    best_cost = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    switches = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    seg_count = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    last_start = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    full_keys = [[()] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    prev_index = [[-1] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    prev_state = [[_NO_SOURCE] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]

    # windows[p][s] contains state-p prefixes to which a source-s segment is
    # appended. The two no-source rows hold only index 0 until it expires.
    windows = tuple(
        tuple(deque([0]) for _s in (SOURCE_A, SOURCE_B))
        for _p in range(3)
    )

    for j in range(1, n + 1):
        low = j - max_len
        for row in windows:
            for window in row:
                while window and window[0] < low:
                    window.popleft()

        for s in (SOURCE_A, SOURCE_B):
            best_candidate = None
            chosen_predecessor = _NO_SOURCE
            for p in range(3):
                window = windows[p][s]
                if not window:
                    continue
                i = window[0]
                if p < 2:
                    prefix_cost = best_cost[p][i]
                    prefix_switches = switches[p][i]
                    prefix_count = seg_count[p][i]
                    # The new final source is fixed, so a tie in the visible
                    # final-plan keys is resolved with the prefix's own full
                    # continuity order.
                    prefix_tie = full_keys[p][i]
                    predecessor_state = p
                    added_switch = 0 if p == s else 1
                else:
                    prefix_cost = 0
                    prefix_switches = 0
                    prefix_count = 0
                    prefix_tie = ()
                    predecessor_state = _NO_SOURCE
                    added_switch = 0

                candidate = (
                    prefix_cost - prefixes[s][i] + prefixes[s][j] + fees[s],
                    prefix_switches + added_switch,
                    prefix_count + 1,
                    i,
                    s,
                    prefix_tie,
                )
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
                    chosen_predecessor = predecessor_state

            cost, switch_count, count, i, s, _prefix_tie = best_candidate
            best_cost[s][j] = cost
            switches[s][j] = switch_count
            seg_count[s][j] = count
            last_start[s][j] = i
            full_keys[s][j] = best_candidate
            prev_index[s][j] = i
            prev_state[s][j] = chosen_predecessor

            # A real A/B prefix becomes a predecessor only in windows keyed
            # by its actual last-source state.
            for new_s, window in enumerate(windows[s]):
                key = (
                    best_cost[s][j] - prefixes[new_s][j],
                    switches[s][j],
                    seg_count[s][j],
                )
                while window:
                    tail = window[-1]
                    tail_key = (
                        best_cost[s][tail] - prefixes[new_s][tail],
                        switches[s][tail],
                        seg_count[s][tail],
                    )
                    if tail_key <= key:
                        break
                    window.pop()
                window.append(j)

    final_candidate = None
    final_state = SOURCE_A
    for p in (SOURCE_A, SOURCE_B):
        candidate = (
            best_cost[p][n],
            switches[p][n],
            seg_count[p][n],
            last_start[p][n],
            p,
        )
        if final_candidate is None or candidate < final_candidate:
            final_candidate = candidate
            final_state = p

    return _reconstruct_states(
        n, prev_index, prev_state, final_state, final_candidate[0])


def _reconstruct(n: int, prev_index: list[int], prev_source: list[int],
                 total_cost: int) -> Solution:
    segments: list[Segment] = []
    end = n
    while end > 0:
        start = prev_index[end]
        segments.append(Segment(start, end, SOURCE_NAMES[prev_source[end]]))
        end = start
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))


def _reconstruct_states(n: int, prev_index: list[list[int]],
                        prev_state: list[list[int]], final_state: int,
                        total_cost: int) -> Solution:
    segments: list[Segment] = []
    end = n
    state = final_state
    while end > 0:
        start = prev_index[state][end]
        segments.append(Segment(start, end, SOURCE_NAMES[state]))
        next_state = prev_state[state][end]
        end = start
        state = next_state
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))
