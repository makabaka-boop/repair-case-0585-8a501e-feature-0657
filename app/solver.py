"""Sliding-window dynamic program for satellite telemetry gap repair.

Problem
-------
A gap of ``n`` positions ``[0, n)`` must be covered by back-to-back
half-open segments ``[start, end)``. Every segment uses exactly one source
(A or B). Choosing a segment of source ``s`` over ``[i, j)`` costs

    fee[s] + sum(cost[k][s] for k in range(i, j))

so the per-source activation fee is paid once *per segment* that uses the
source. Consecutive segments of the same source are therefore never free:
a new segment always pays the fee again.

Recurrence
----------
``dp[j]`` is the lexicographically best solution covering ``[0, j)``:

    dp[0] = (cost=0, segments=0, prev=-1, source=-1)
    dp[j] = min over source s in (A, B), i in [j-L, j-1], i >= 0
            (dp[i].cost + fee[s] + prefix_s[j] - prefix_s[i],
             dp[i].segments + 1, i, s)

Candidates are compared strictly in the order
``(cost, segments, prev index, source)`` with source A < B, so the
recurrence determines a unique plan; there is no arbitrary tie breaking.

For a fixed source the innermost term is ``dp[i].cost - prefix_s[i]`` plus
quantities independent of ``i``, and the feasible predecessors form the
sliding window ``[j-L, j-1]``. Two monotonic deques maintain the minimum
of

    (dp[i].cost - prefix_s[i], dp[i].segments, i)

over that window in O(1) amortised time per endpoint, giving O(n) time
(which satisfies the O(n log n) requirement) and O(n) memory. No solver
is invoked and the L possible starts of an endpoint are never enumerated.
"""

from collections import deque
from dataclasses import dataclass

# Source ids double as tie-breakers: A < B.
SOURCE_A = 0
SOURCE_B = 1
SOURCE_NAMES = ("A", "B")


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
          max_len: int) -> Solution:
    """Compute the optimal cover of [0, n).

    ``costs[k]`` is the pair of per-position costs ``(cost of A, cost of
    B)`` at position ``k``. All inputs are assumed already validated by the
    API layer; the algorithm itself trusts the bounds.
    """
    fees = (fee_a, fee_b)

    # Prefix sums per source: prefix_s[j] = sum of source-s costs on [0,j).
    prefix_a = [0] * (n + 1)
    prefix_b = [0] * (n + 1)
    for k in range(n):
        a, b = costs[k]
        prefix_a[k + 1] = prefix_a[k] + a
        prefix_b[k + 1] = prefix_b[k] + b
    prefixes = (prefix_a, prefix_b)

    # DP tables.
    best_cost = [0] * (n + 1)
    best_seg_count = [0] * (n + 1)
    prev_index = [-1] * (n + 1)
    prev_source = [-1] * (n + 1)

    # One monotonic deque per source. Each entry is index i with key
    # (best_cost[i] - prefix_s[i], best_seg_count[i], i). Keys along the
    # deque are non-decreasing; equal keys keep the smaller index at the
    # front (we only pop on strictly-greater keys), matching the
    # predecessor-index tie breaker.
    #
    # Inserting i into deque s is only legal for endpoints j > i, so the
    # seed i=0 is inserted before processing endpoint 1.
    windows: tuple[deque, deque] = (deque([0]), deque([0]))

    for j in range(1, n + 1):
        low = j - max_len  # smallest predecessor still inside the window

        # Evict indices that left the window [j-L, j-1]. The predecessor
        # i == j-L is still feasible (segment of length exactly L), so only
        # strictly smaller indices are removed.
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

        # j becomes a feasible predecessor for endpoints j+1 .. j+L.
        for s, window in enumerate(windows):
            key = (best_cost[j] - prefixes[s][j], best_seg_count[j])
            while window:
                tail = window[-1]
                tail_key = (
                    best_cost[tail] - prefixes[s][tail],
                    best_seg_count[tail],
                )
                # Strictly worse (greater key) tails are dominated forever;
                # equal keys are kept because their smaller index must win
                # the predecessor-index tie breaker and expires earlier.
                if tail_key <= key:
                    break
                window.pop()
            window.append(j)

    # Reconstruct the unique optimal plan back to front.
    segments: list[Segment] = []
    j = n
    while j > 0:
        i = prev_index[j]
        segments.append(Segment(i, j, SOURCE_NAMES[prev_source[j]]))
        j = i
    segments.reverse()

    return Solution(cost=best_cost[n], segments=tuple(segments))
