"""Sliding-window dynamic programs for satellite telemetry gap repair.

Problem
-------
A gap of ``n`` positions ``[0, n)`` must be covered by back-to-back
half-open segments ``[start, end)``. Every segment uses exactly one source
(A or B). Choosing a segment of source ``s`` over ``[i, j)`` costs

    fee[s] + sum(cost[k][s] for k in range(i, j))

so the per-source activation fee is paid once *per segment* that uses the
source. Consecutive segments of the same source are therefore never free:
a new segment always pays the fee again.

Two objectives share the cost model:

``default``
    The lexicographic order is
    ``(cost, segments, predecessor index, source)`` with source A < B.

``continuity``
    Cost is still minimized first; among minimum-cost plans the solver then
    minimizes, in order, the number of adjacent source switches (the first
    segment does not count as a switch), the segment count, the start of
    the last segment and the source of the last segment (A < B). Prefix
    plans are adjudicated by the same order recursively: when two plans
    covering the same endpoint are tied through their last segment, their
    common-length prefixes are compared in that order.

Default recurrence
------------------
``dp[j]`` is the lexicographically best solution covering ``[0, j)``:

    dp[0] = (cost=0, segments=0, prev=-1, source=-1)
    dp[j] = min over source s in (A, B), i in [j-L, j-1], i >= 0
            (dp[i].cost + fee[s] + prefix_s[j] - prefix_s[i],
             dp[i].segments + 1, i, s)

For a fixed source the innermost term is ``dp[i].cost - prefix_s[i]`` plus
quantities independent of ``i``, and the feasible predecessors form the
sliding window ``[j-L, j-1]``. Two monotonic deques maintain the minimum
of

    (dp[i].cost - prefix_s[i], dp[i].segments, i)

over that window in O(1) amortised time per endpoint.

Continuity recurrence
---------------------
The continuity mode keeps, per endpoint ``j``, one best prefix per source
of its last segment: ``dp[j][p]`` for ``p in (A, B)``. A source-less
sentinel ``dp[0] = (0, 0, 0)`` (cost, switches, segments) seeds first
segments; it is never stored in a window and a first segment never pays a
switch.

For a new segment of source ``s`` extending prefix ``(i, p)`` the switch
indicator is ``p != s``; it only shifts the switch component by a
per-deque constant (0 or 1) and therefore does not affect the monotonic
window ordering. Windows are maintained per (previous source, new source)
pair ``(p, s)`` over the constant-independent key

    (dp[i][p].cost - prefix_s[i],
     dp[i][p].switches,
     dp[i][p].segments)

and the candidate ending at ``j`` is compared as

    (cost, switches + (p != s), segments + 1, predecessor i, prefix rank)

where the *prefix rank* is the predecessor plan adjudicated in the very
same continuity order,
``(dp[i][p].cost, dp[i][p].switches, dp[i][p].segments,
   dp[i][p].prev, p)``. It is only consulted when cost, switches,
segments and the last-segment start all tie (which forces equal
predecessor indices); the first-segment candidate carries the rank of
the empty prefix. Note the window key's switch term already includes
the new edge, so two equal-looking candidates may rest on prefixes with
different switch counts -- the prefix rank, not the edge-adjusted
value, performs the recursive tie break. At the final endpoint the
last comparison component is the candidate's own last source instead,
implementing the "last segment source" tie breaker.

Both modes run in O(n) time and O(n) memory: every index enters and
leaves each deque at most once. The continuity tables and deques are
disjoint from the default mode's state; backtracking only reads the mode
that was solved, so continuity can never rewrite a prefix the default
mode had discarded, and default tie results are unchanged.
"""

from collections import deque
from dataclasses import dataclass

# Source ids double as tie-breakers: A < B.
SOURCE_A = 0
SOURCE_B = 1
SOURCE_NAMES = ("A", "B")

# Objectives.
OBJECTIVE_DEFAULT = "default"
OBJECTIVE_CONTINUITY = "continuity"
OBJECTIVES = (OBJECTIVE_DEFAULT, OBJECTIVE_CONTINUITY)

# Source-less predecessor of every first segment.
NO_SOURCE = -1


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
    B)`` at position ``k``. All inputs are assumed already validated by the
    API layer; the algorithm itself trusts the bounds. ``objective`` is
    either ``"default"`` or ``"continuity"``.
    """
    if objective == OBJECTIVE_CONTINUITY:
        return _solve_continuity(n, costs, fee_a, fee_b, max_len)
    return _solve_default(n, costs, fee_a, fee_b, max_len)


def _prefix_sums(n, costs):
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

    # Prefix sums per source: prefix_s[j] = sum of source-s costs on [0,j).
    prefix_a, prefix_b = _prefix_sums(n, costs)
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


def _solve_continuity(n: int, costs: list[tuple[int, int]], fee_a: int,
                      fee_b: int, max_len: int) -> Solution:
    """Minimize (cost, switches, segments, last start, last source).

    Tables are flat, index ``2*j + p`` holding the best prefix covering
    ``[0, j)`` whose last segment uses source ``p``.
    """
    fees = (fee_a, fee_b)
    prefix_a, prefix_b = _prefix_sums(n, costs)
    prefixes = (prefix_a, prefix_b)

    size = 2 * (n + 1)
    best_cost = [0] * size
    switches = [0] * size
    seg_count = [0] * size
    prev_index = [-1] * size
    prev_source = [-1] * size

    # windows[p][s]: monotonic deque of indices i whose best prefix ends in
    # source p and which may be extended by a new segment of source s.
    # Entry key (the (p != s) switch edge is a per-deque constant and is
    # left out):
    #   (best_cost[i][p] - prefix_s[i], switches[i][p], seg_count[i][p])
    # Keys are non-decreasing; equal keys keep the smaller index at the
    # front (last-segment-start tie breaker), as in the default mode.
    #
    # The source-less sentinel i=0 is never inserted: first segments are
    # offered as explicit candidates while j <= L.
    windows = ((deque(), deque()), (deque(), deque()))

    for j in range(1, n + 1):
        low = j - max_len
        for p in (SOURCE_A, SOURCE_B):
            for s in (SOURCE_A, SOURCE_B):
                window = windows[p][s]
                while window and window[0] < low:
                    window.popleft()

        for s in (SOURCE_A, SOURCE_B):
            # First-segment candidate from the source-less sentinel; its
            # prefix rank is the empty prefix (0, 0, 0, -1, -1). i == 0
            # only occurs here, since index 0 is never enqueued, so it can
            # never tie against a window candidate on
            # (..., i, prefix rank).
            best_candidate = None
            if j <= max_len:
                best_candidate = (
                    prefixes[s][j] + fees[s],
                    0,  # first segment pays no switch
                    1,
                    0,
                    (0, 0, 0, NO_SOURCE, NO_SOURCE),
                )

            for p in (SOURCE_A, SOURCE_B):
                window = windows[p][s]
                if not window:
                    continue
                i = window[0]
                k = 2 * i + p
                candidate = (
                    best_cost[k] - prefixes[s][i] + prefixes[s][j]
                    + fees[s],
                    switches[k] + (1 if p != s else 0),
                    seg_count[k] + 1,
                    i,
                    (best_cost[k], switches[k], seg_count[k],
                     prev_index[k], p),
                )
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate

            cost, sw, count, i, rank = best_candidate
            # The predecessor source is the last component of the prefix
            # rank (NO_SOURCE for the sentinel); do not read the stale
            # candidate-loop variable p.
            p = rank[-1]
            out = 2 * j + s
            best_cost[out] = cost
            switches[out] = sw
            seg_count[out] = count
            prev_index[out] = i
            prev_source[out] = p

        # State (j, s) becomes a feasible predecessor for endpoints
        # j+1 .. j+L under a following segment of either source.
        for p in (SOURCE_A, SOURCE_B):
            k = 2 * j + p
            for s in (SOURCE_A, SOURCE_B):
                window = windows[p][s]
                key = (
                    best_cost[k] - prefixes[s][j],
                    switches[k],
                    seg_count[k],
                )
                while window:
                    tail = window[-1]
                    tk = 2 * tail + p
                    tail_key = (
                        best_cost[tk] - prefixes[s][tail],
                        switches[tk],
                        seg_count[tk],
                    )
                    # Equal keys stay: the smaller index wins the
                    # last-segment-start tie breaker and expires earlier.
                    if tail_key <= key:
                        break
                    window.pop()
                window.append(j)

    # Final adjudication: the last comparison component is the plan's own
    # last source (A < B), not its predecessor's source.
    final_candidates = []
    for p in (SOURCE_A, SOURCE_B):
        k = 2 * n + p
        final_candidates.append(
            (best_cost[k], switches[k], seg_count[k], prev_index[k], p))
    cost, _, _, _, last_source = min(final_candidates)

    # Backtrack exclusively through continuity-mode state.
    segments: list[Segment] = []
    j = n
    p = last_source
    while j > 0:
        k = 2 * j + p
        i = prev_index[k]
        segments.append(Segment(i, j, SOURCE_NAMES[p]))
        p = prev_source[k]
        j = i
    segments.reverse()

    return Solution(cost=cost, segments=tuple(segments))
