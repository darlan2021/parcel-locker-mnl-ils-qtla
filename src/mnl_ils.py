from __future__ import annotations

"""
Problem-specific ILS/GRASP/VND/Path-Relinking heuristic for the MNL
parcel-locker location/redesign problem.

Model represented by this solver
--------------------------------
Let z = (r, x), where:
    r_k = 1 if existing POP station k remains open, k in K
    x_j = 1 if candidate locker j is opened, j in J

The MNL objective is
    F(z) = sum_i (sum_m a[i,m] z_m) / (sum_m b[i,m] z_m)

with CData's column order
    [existing POP stations | candidate lockers].

Cardinality requirements:
    sum_j x_j <= p
    sum_k r_k >= n_k - p

The local search evaluates the penalized objective
    F_pen(z) = F(z)
               - rho_C * (v_C + gamma*v_C^2)
               - rho_E * (v_E + gamma*v_E^2)

where
    v_C = max(0, sum_j x_j - p)
    v_E = max(0, (n_k - p) - sum_k r_k).

Neighborhoods
-------------
A  : controlled single add/drop. May visit nearby infeasible configurations.
     Reverse add/drop decisions are tabu for tau_tabu accepted A-moves/VND moves,
     unless aspiration applies.

S1 : one-for-one swap, performed inside either K or J. It preserves the two
     group cardinalities and is used only from feasible states.

S2 : two-for-two swap. It is built from either two within-group swaps or one
     swap in K plus one swap in J. It also preserves group cardinalities and
     is used only from feasible states.

The implementation uses incremental numerator/denominator updates, so one
move evaluation costs O(|I|), not O(|I|*|J|).

Example
-------
    from data import CData
    from mnl_ils import ILSConfig, MNLParcelLockerILS

    instance = CData.from_total_nodes(N=100, p=10, lambda_=1.0, seed=1)
    cfg = ILSConfig(seed=123, max_iterations=250)
    solver = MNLParcelLockerILS(instance, cfg)
    result = solver.solve()
    print(result)

The uploaded CData class is used directly: this solver consumes instance.a
and instance.b, which CData builds from the generated service, demand and MNL
attractiveness data.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence
import argparse
import math
import time

import numpy as np


_EPS = 1e-12


@dataclass(slots=True)
class ILSConfig:
    # GRASP
    grasp_starts: int = 8
    grasp_alpha: float = 0.20

    # Local search / tabu
    max_vnd_passes: int = 250
    tau_tabu: int = 7
    max_cardinality_violation: int = 2
    s2_sample_size: int = 350
    first_improvement: bool = False

    # Component switches (useful for ablation/component analysis).
    enable_A: bool = True
    enable_S1: bool = True
    enable_S2: bool = True
    enable_path_relink: bool = True

    # Penalization. If penalty_weight is None it is scaled from the first
    # feasible GRASP objective.
    penalty_weight: Optional[float] = None
    penalty_multiplier: float = 1.50
    penalty_quadratic: float = 0.35

    # ILS
    max_iterations: int = 300
    time_limit: Optional[float] = None
    initial_perturbation_strength: int = 2
    min_perturbation_strength: int = 1
    max_perturbation_strength: int = 10
    stagnation_window: int = 12
    acceptance_tolerance: float = 0.01

    # Adaptive perturbation base probabilities
    perturb_prob_A: float = 0.15
    perturb_prob_S1: float = 0.55
    perturb_prob_S2: float = 0.30

    # Elite set / Path Relinking
    elite_size: int = 8
    elite_min_hamming: int = 2
    path_relink_interval: int = 10
    path_relink_bidirectional: bool = True
    path_relink_s2_samples: int = 80

    seed: int = 0
    verbose: bool = True


@dataclass
class State:
    y: np.ndarray
    num: np.ndarray
    den: np.ndarray
    raw: float
    penalized: float = -math.inf

    def copy(self) -> "State":
        return State(
            y=self.y.copy(),
            num=self.num.copy(),
            den=self.den.copy(),
            raw=float(self.raw),
            penalized=float(self.penalized),
        )


@dataclass
class SearchProfile:
    """Timing and activity counters collected during one solve()."""

    # Exclusive major components. Their sum plus ``other_seconds`` is close to
    # total_seconds. The A/S1/S2/repair timings below are nested inside
    # local_search_seconds and therefore must not be added again.
    total_seconds: float = 0.0
    grasp_seconds: float = 0.0
    local_search_seconds: float = 0.0
    perturbation_seconds: float = 0.0
    path_relink_seconds: float = 0.0  # PR core only; post-PR VND is LS time.

    # Detailed VND timings (subsets of local_search_seconds).
    A_seconds: float = 0.0
    S1_seconds: float = 0.0
    S2_seconds: float = 0.0
    repair_seconds: float = 0.0

    # Calls.
    grasp_calls: int = 0
    local_search_calls: int = 0
    perturbation_calls: int = 0
    path_relink_calls: int = 0
    repair_calls: int = 0
    A_calls: int = 0
    S1_calls: int = 0
    S2_calls: int = 0

    # Move evaluations / accepted moves.
    grasp_move_evaluations: int = 0
    grasp_steps: int = 0
    A_move_evaluations: int = 0
    S1_move_evaluations: int = 0
    S2_move_evaluations: int = 0
    A_accepted: int = 0
    S1_accepted: int = 0
    S2_accepted: int = 0
    repair_moves: int = 0

    # Perturbation operator usage.
    perturb_A_moves: int = 0
    perturb_S1_moves: int = 0
    perturb_S2_moves: int = 0

    # Path-relinking effort.
    PR_move_evaluations: int = 0
    PR_steps: int = 0
    PR_A_steps: int = 0
    PR_S1_steps: int = 0
    PR_S2_steps: int = 0

    def copy(self) -> "SearchProfile":
        return SearchProfile(**self.__dict__)

    @property
    def major_accounted_seconds(self) -> float:
        return (
            self.grasp_seconds
            + self.local_search_seconds
            + self.perturbation_seconds
            + self.path_relink_seconds
        )

    @property
    def other_seconds(self) -> float:
        return max(0.0, self.total_seconds - self.major_accounted_seconds)

    @property
    def local_search_other_seconds(self) -> float:
        return max(
            0.0,
            self.local_search_seconds
            - self.A_seconds
            - self.S1_seconds
            - self.S2_seconds
            - self.repair_seconds,
        )

    def to_dict(self) -> dict[str, float | int]:
        d = dict(self.__dict__)
        d["other_seconds"] = self.other_seconds
        d["local_search_other_seconds"] = self.local_search_other_seconds
        return d

    def format_report(self) -> str:
        total = max(self.total_seconds, _EPS)
        ls = max(self.local_search_seconds, _EPS)

        def major_line(label: str, value: float) -> str:
            return f"  {label:<25s} {value:10.4f} s  ({100.0 * value / total:6.2f}%)"

        def ls_line(label: str, value: float) -> str:
            return f"  {label:<25s} {value:10.4f} s  ({100.0 * value / ls:6.2f}% of LS)"

        lines = [
            "Computational profile",
            "---------------------",
            f"Total wall time            {self.total_seconds:10.4f} s",
            "Major components (exclusive):",
            major_line("GRASP construction", self.grasp_seconds),
            major_line("VND / local search", self.local_search_seconds),
            major_line("Adaptive perturbation", self.perturbation_seconds),
            major_line("Path Relinking core", self.path_relink_seconds),
            major_line("Other overhead", self.other_seconds),
            "",
            "VND / local-search detail (nested inside local-search time):",
            ls_line("A add/drop", self.A_seconds),
            ls_line("S1 one-for-one", self.S1_seconds),
            ls_line("S2 two-for-two", self.S2_seconds),
            ls_line("Feasibility repair", self.repair_seconds),
            ls_line("VND orchestration/other", self.local_search_other_seconds),
            "",
            "Effort counters:",
            f"  GRASP: calls={self.grasp_calls}, evals={self.grasp_move_evaluations}, steps={self.grasp_steps}",
            f"  A:     calls={self.A_calls}, evals={self.A_move_evaluations}, accepted={self.A_accepted}",
            f"  S1:    calls={self.S1_calls}, evals={self.S1_move_evaluations}, accepted={self.S1_accepted}",
            f"  S2:    calls={self.S2_calls}, evals={self.S2_move_evaluations}, accepted={self.S2_accepted}",
            f"  Repair: calls={self.repair_calls}, moves={self.repair_moves}",
            f"  Perturbation: calls={self.perturbation_calls}, A={self.perturb_A_moves}, S1={self.perturb_S1_moves}, S2={self.perturb_S2_moves}",
            f"  Path Relinking: calls={self.path_relink_calls}, evals={self.PR_move_evaluations}, steps={self.PR_steps} "
            f"(A={self.PR_A_steps}, S1={self.PR_S1_steps}, S2={self.PR_S2_steps})",
            "",
            "Note: Path Relinking time is the relinking core only. Its final VND",
            "intensification is counted under VND/local search, avoiding double counting.",
        ]
        return "\n".join(lines)


@dataclass
class SearchResult:
    objective: float
    penalized_objective: float
    feasible: bool
    y: np.ndarray
    open_existing: list[int]
    closed_existing: list[int]
    open_candidates: list[int]
    iterations: int
    elapsed_seconds: float
    perturbation_strength: int
    profile: SearchProfile
    history: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"SearchResult(objective={self.objective:.8f}, "
            f"feasible={self.feasible}, "
            f"open_candidates={self.open_candidates}, "
            f"closed_existing={self.closed_existing}, "
            f"iterations={self.iterations}, "
            f"elapsed={self.elapsed_seconds:.3f}s)"
        )

    def profile_report(self) -> str:
        return self.profile.format_report()


@dataclass(frozen=True)
class Move:
    kind: str
    remove: tuple[int, ...] = ()
    add: tuple[int, ...] = ()

    @property
    def changed(self) -> tuple[int, ...]:
        return self.remove + self.add


class MNLParcelLockerILS:
    """GRASP + ILS + VND + tabu add/drop + adaptive perturbation + PR."""

    def __init__(self, data, config: Optional[ILSConfig] = None):
        self.data = data
        self.cfg = config or ILSConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        # CData concatenates [existing | candidate].
        self.a = np.asarray(data.a, dtype=float)
        self.b = np.asarray(data.b, dtype=float)
        self.n_i = int(data.n_i)
        self.n_k = int(data.n_k)
        self.n_j = int(data.n_j)
        self.p = int(data.p)
        self.n = self.n_k + self.n_j

        if self.a.shape != (self.n_i, self.n):
            raise ValueError(
                f"Expected a.shape={(self.n_i, self.n)}, got {self.a.shape}."
            )
        if self.b.shape != (self.n_i, self.n):
            raise ValueError(
                f"Expected b.shape={(self.n_i, self.n)}, got {self.b.shape}."
            )
        if np.any(self.b <= 0.0) or not np.all(np.isfinite(self.b)):
            raise ValueError("All MNL b coefficients must be finite and positive.")
        if not np.all(np.isfinite(self.a)):
            raise ValueError("All a coefficients must be finite.")

        self.K = np.arange(0, self.n_k, dtype=int)
        self.J = np.arange(self.n_k, self.n, dtype=int)

        # Set after the first GRASP construction unless explicitly supplied.
        self.rho = (
            float(self.cfg.penalty_weight)
            if self.cfg.penalty_weight is not None
            else None
        )

        # Tabu memory for A moves. Entry stores the iteration through which
        # the reverse action is forbidden.
        self.tabu_drop_until = np.full(self.n, -1, dtype=int)
        self.tabu_add_until = np.full(self.n, -1, dtype=int)
        self.move_clock = 0

        self.best_feasible: Optional[State] = None
        self.elite: list[State] = []
        self.history: list[dict] = []
        self.profile = SearchProfile()

    # ------------------------------------------------------------------
    # Objective, feasibility and incremental move evaluation
    # ------------------------------------------------------------------
    def _make_state(self, y: np.ndarray) -> State:
        y = np.asarray(y, dtype=np.int8)
        num = self.a @ y
        den = self.b @ y
        raw = self._ratio_sum(num, den)
        st = State(y=y.copy(), num=num, den=den, raw=raw)
        st.penalized = self._penalized(st.raw, st.y)
        return st

    @staticmethod
    def _ratio_sum(num: np.ndarray, den: np.ndarray) -> float:
        if np.any(den <= _EPS):
            return -math.inf
        value = np.sum(num / den)
        return float(value) if np.isfinite(value) else -math.inf

    def counts(self, y: np.ndarray) -> tuple[int, int]:
        existing_open = int(np.sum(y[: self.n_k]))
        candidate_open = int(np.sum(y[self.n_k :]))
        return existing_open, candidate_open

    def violations(self, y: np.ndarray) -> tuple[int, int]:
        existing_open, candidate_open = self.counts(y)
        v_c = max(0, candidate_open - self.p)
        v_e = max(0, (self.n_k - self.p) - existing_open)
        return v_c, v_e

    def violation_total(self, y: np.ndarray) -> int:
        return sum(self.violations(y))

    def feasible(self, y: np.ndarray) -> bool:
        return self.violation_total(y) == 0

    def _penalized(self, raw: float, y: np.ndarray) -> float:
        if not np.isfinite(raw):
            return -math.inf
        if self.rho is None:
            # During the first GRASP before scaling rho, feasible states have
            # no violation anyway.
            return raw
        vc, ve = self.violations(y)
        g = self.cfg.penalty_quadratic
        penalty = self.rho * ((vc + g * vc * vc) + (ve + g * ve * ve))
        return raw - penalty

    def _move_value(self, state: State, move: Move) -> tuple[float, float]:
        """Return (raw, penalized) after an incremental move."""
        dnum = np.zeros(self.n_i, dtype=float)
        dden = np.zeros(self.n_i, dtype=float)

        if move.remove:
            idx = np.fromiter(move.remove, dtype=int)
            dnum -= np.sum(self.a[:, idx], axis=1)
            dden -= np.sum(self.b[:, idx], axis=1)
        if move.add:
            idx = np.fromiter(move.add, dtype=int)
            dnum += np.sum(self.a[:, idx], axis=1)
            dden += np.sum(self.b[:, idx], axis=1)

        raw = self._ratio_sum(state.num + dnum, state.den + dden)
        if not np.isfinite(raw):
            return -math.inf, -math.inf

        y2 = state.y.copy()
        if move.remove:
            y2[list(move.remove)] = 0
        if move.add:
            y2[list(move.add)] = 1
        return raw, self._penalized(raw, y2)

    def _apply_move(
        self,
        state: State,
        move: Move,
        raw: Optional[float] = None,
        penalized: Optional[float] = None,
        update_tabu: bool = False,
    ) -> State:
        st = state.copy()

        if move.remove:
            idx = np.fromiter(move.remove, dtype=int)
            st.num -= np.sum(self.a[:, idx], axis=1)
            st.den -= np.sum(self.b[:, idx], axis=1)
            st.y[idx] = 0
        if move.add:
            idx = np.fromiter(move.add, dtype=int)
            st.num += np.sum(self.a[:, idx], axis=1)
            st.den += np.sum(self.b[:, idx], axis=1)
            st.y[idx] = 1

        st.raw = self._ratio_sum(st.num, st.den) if raw is None else float(raw)
        st.penalized = (
            self._penalized(st.raw, st.y)
            if penalized is None
            else float(penalized)
        )

        if update_tabu and move.kind == "A":
            self.move_clock += 1
            until = self.move_clock + self.cfg.tau_tabu
            for j in move.add:
                # Reverse drop is tabu.
                self.tabu_drop_until[j] = until
            for j in move.remove:
                # Reverse re-addition is tabu.
                self.tabu_add_until[j] = until

        return st

    def _controlled(self, y: np.ndarray) -> bool:
        vc, ve = self.violations(y)
        q = self.cfg.max_cardinality_violation
        return vc <= q and ve <= q

    def _is_tabu_A(self, state: State, move: Move, resulting_raw: float) -> bool:
        assert move.kind == "A"
        tabu = False
        for j in move.add:
            if self.move_clock <= self.tabu_add_until[j]:
                tabu = True
        for j in move.remove:
            if self.move_clock <= self.tabu_drop_until[j]:
                tabu = True

        if not tabu:
            return False

        # Aspiration: a tabu A move is allowed if it creates a feasible
        # solution strictly better than the best feasible solution so far.
        y2 = state.y.copy()
        if move.remove:
            y2[list(move.remove)] = 0
        if move.add:
            y2[list(move.add)] = 1

        incumbent = (
            -math.inf if self.best_feasible is None else self.best_feasible.raw
        )
        return not (self.feasible(y2) and resulting_raw > incumbent + 1e-12)

    # ------------------------------------------------------------------
    # GRASP construction
    # ------------------------------------------------------------------
    def grasp_construct(self) -> State:
        """
        Build a boundary-feasible P-for-P redesign:
            p candidates opened and p existing stations closed.

        At each construction step, all add-candidate/drop-existing pairs are
        scored by the exact MNL objective, then one move is sampled from a
        GRASP restricted candidate list.
        """
        self.profile.grasp_calls += 1
        t0 = time.perf_counter()
        try:
            y = np.zeros(self.n, dtype=np.int8)
            y[: self.n_k] = 1
            st = self._make_state(y)

            for _ in range(self.p):
                closed_cand = np.flatnonzero(st.y[self.n_k :] == 0) + self.n_k
                open_exist = np.flatnonzero(st.y[: self.n_k] == 1)

                scored: list[tuple[float, Move]] = []
                for c in closed_cand:
                    for k in open_exist:
                        mv = Move("GRASP", remove=(int(k),), add=(int(c),))
                        raw, _ = self._move_value(st, mv)
                        self.profile.grasp_move_evaluations += 1
                        scored.append((raw, mv))

                if not scored:
                    break

                vals = np.asarray([q[0] for q in scored], dtype=float)
                best = float(np.max(vals))
                worst = float(np.min(vals))
                threshold = best - self.cfg.grasp_alpha * (best - worst)
                rcl = [scored[q] for q in np.flatnonzero(vals >= threshold)]
                raw, mv = rcl[int(self.rng.integers(len(rcl)))]
                st = self._apply_move(st, mv, raw=raw)
                self.profile.grasp_steps += 1

            if not self.feasible(st.y):
                raise RuntimeError("GRASP construction failed to produce a feasible state.")
            return st
        finally:
            self.profile.grasp_seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Neighborhood A: controlled add/drop with tabu and aspiration
    # ------------------------------------------------------------------
    def _best_A_move(self, state: State) -> Optional[tuple[Move, float, float]]:
        self.profile.A_calls += 1
        t0 = time.perf_counter()
        try:
            best_tuple = None
            best_pen = state.penalized

            for j in range(self.n):
                if state.y[j]:
                    mv = Move("A", remove=(j,))
                else:
                    mv = Move("A", add=(j,))

                y2 = state.y.copy()
                if mv.remove:
                    y2[j] = 0
                else:
                    y2[j] = 1

                if not self._controlled(y2):
                    continue

                raw, pen = self._move_value(state, mv)
                self.profile.A_move_evaluations += 1
                if self._is_tabu_A(state, mv, raw):
                    continue

                if pen > best_pen + 1e-12:
                    best_tuple = (mv, raw, pen)
                    best_pen = pen
                    if self.cfg.first_improvement:
                        return best_tuple

            return best_tuple
        finally:
            self.profile.A_seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Neighborhood S1: one-for-one swaps
    # ------------------------------------------------------------------
    def _best_S1_move(self, state: State) -> Optional[tuple[Move, float, float]]:
        self.profile.S1_calls += 1
        t0 = time.perf_counter()
        try:
            if not self.feasible(state.y):
                return None

            best_tuple = None
            best_pen = state.penalized

            for group in (self.K, self.J):
                open_idx = group[state.y[group] == 1]
                closed_idx = group[state.y[group] == 0]
                for r in open_idx:
                    for a in closed_idx:
                        mv = Move("S1", remove=(int(r),), add=(int(a),))
                        raw, pen = self._move_value(state, mv)
                        self.profile.S1_move_evaluations += 1
                        if pen > best_pen + 1e-12:
                            best_tuple = (mv, raw, pen)
                            best_pen = pen
                            if self.cfg.first_improvement:
                                return best_tuple

            return best_tuple
        finally:
            self.profile.S1_seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Neighborhood S2: two-for-two swaps, sampled to control complexity
    # ------------------------------------------------------------------
    def _sample_S2_move(self, state: State) -> Optional[Move]:
        if not self.feasible(state.y):
            return None

        open_k = self.K[state.y[self.K] == 1]
        closed_k = self.K[state.y[self.K] == 0]
        open_j = self.J[state.y[self.J] == 1]
        closed_j = self.J[state.y[self.J] == 0]

        patterns = []
        if len(open_k) >= 2 and len(closed_k) >= 2:
            patterns.append("KK")
        if len(open_j) >= 2 and len(closed_j) >= 2:
            patterns.append("JJ")
        if len(open_k) >= 1 and len(closed_k) >= 1 and len(open_j) >= 1 and len(closed_j) >= 1:
            patterns.append("KJ")

        if not patterns:
            return None

        pat = patterns[int(self.rng.integers(len(patterns)))]
        if pat == "KK":
            r = self.rng.choice(open_k, size=2, replace=False)
            a = self.rng.choice(closed_k, size=2, replace=False)
        elif pat == "JJ":
            r = self.rng.choice(open_j, size=2, replace=False)
            a = self.rng.choice(closed_j, size=2, replace=False)
        else:
            r = np.array(
                [self.rng.choice(open_k), self.rng.choice(open_j)], dtype=int
            )
            a = np.array(
                [self.rng.choice(closed_k), self.rng.choice(closed_j)], dtype=int
            )

        return Move(
            "S2",
            remove=tuple(int(x) for x in r),
            add=tuple(int(x) for x in a),
        )

    def _best_S2_move(self, state: State) -> Optional[tuple[Move, float, float]]:
        self.profile.S2_calls += 1
        t0 = time.perf_counter()
        try:
            if not self.feasible(state.y):
                return None

            best_tuple = None
            best_pen = state.penalized
            seen = set()

            for _ in range(self.cfg.s2_sample_size):
                mv = self._sample_S2_move(state)
                if mv is None:
                    break
                key = (tuple(sorted(mv.remove)), tuple(sorted(mv.add)))
                if key in seen:
                    continue
                seen.add(key)

                raw, pen = self._move_value(state, mv)
                self.profile.S2_move_evaluations += 1
                if pen > best_pen + 1e-12:
                    best_tuple = (mv, raw, pen)
                    best_pen = pen
                    if self.cfg.first_improvement:
                        return best_tuple

            return best_tuple
        finally:
            self.profile.S2_seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Repair and VND local search
    # ------------------------------------------------------------------
    def _repair_to_feasible(self, state: State) -> State:
        """
        Monotone feasibility repair used after perturbation if the penalized
        local search stops in a nearby infeasible state.
        """
        self.profile.repair_calls += 1
        t0 = time.perf_counter()
        try:
            st = state.copy()

            guard = 0
            while not self.feasible(st.y):
                guard += 1
                if guard > 2 * self.n + 10:
                    raise RuntimeError("Feasibility repair did not converge.")

                vc, ve = self.violations(st.y)
                candidates: list[tuple[int, float, Move, float]] = []

                if vc > 0:
                    for j in self.J[st.y[self.J] == 1]:
                        mv = Move("A", remove=(int(j),))
                        raw, pen = self._move_value(st, mv)
                        y2 = st.y.copy()
                        y2[j] = 0
                        candidates.append((self.violation_total(y2), -raw, mv, pen))

                if ve > 0:
                    for k in self.K[st.y[self.K] == 0]:
                        mv = Move("A", add=(int(k),))
                        raw, pen = self._move_value(st, mv)
                        y2 = st.y.copy()
                        y2[k] = 1
                        candidates.append((self.violation_total(y2), -raw, mv, pen))

                if not candidates:
                    raise RuntimeError("No repair move exists.")

                candidates.sort(key=lambda x: (x[0], x[1]))
                _, neg_raw, mv, pen = candidates[0]
                st = self._apply_move(
                    st, mv, raw=-neg_raw, penalized=pen, update_tabu=True
                )
                self.profile.repair_moves += 1

            return st
        finally:
            self.profile.repair_seconds += time.perf_counter() - t0

    def local_search(self, state: State) -> State:
        """
        Variable-neighborhood descent over enabled neighborhoods. By default:
            A -> S1 -> S2, restarting at A after every accepted improvement.

        S1/S2 are called only from feasible states. A can cross the cardinality
        boundary within max_cardinality_violation.
        """
        self.profile.local_search_calls += 1
        t0 = time.perf_counter()
        try:
            st = state.copy()
            st.penalized = self._penalized(st.raw, st.y)

            neighborhoods = []
            if self.cfg.enable_A:
                neighborhoods.append("A")
            if self.cfg.enable_S1:
                neighborhoods.append("S1")
            if self.cfg.enable_S2:
                neighborhoods.append("S2")

            k = 0
            passes = 0

            while k < len(neighborhoods) and passes < self.cfg.max_vnd_passes:
                passes += 1
                name = neighborhoods[k]

                if name == "A":
                    candidate = self._best_A_move(st)
                elif name == "S1":
                    candidate = self._best_S1_move(st)
                else:
                    candidate = self._best_S2_move(st)

                if candidate is None:
                    k += 1
                    continue

                mv, raw, pen = candidate
                st = self._apply_move(
                    st,
                    mv,
                    raw=raw,
                    penalized=pen,
                    update_tabu=(mv.kind == "A"),
                )
                if mv.kind == "A":
                    self.profile.A_accepted += 1
                elif mv.kind == "S1":
                    self.profile.S1_accepted += 1
                elif mv.kind == "S2":
                    self.profile.S2_accepted += 1
                self._register_best(st)
                k = 0

            if not self.feasible(st.y):
                st = self._repair_to_feasible(st)

                # Compact feasible intensification after repair.
                getters = []
                if self.cfg.enable_S1:
                    getters.append(self._best_S1_move)
                if self.cfg.enable_S2:
                    getters.append(self._best_S2_move)

                improved = True
                rounds = 0
                while improved and rounds < 25 and getters:
                    rounds += 1
                    improved = False
                    for getter in getters:
                        cand = getter(st)
                        if cand is not None:
                            mv, raw, pen = cand
                            st = self._apply_move(st, mv, raw=raw, penalized=pen)
                            if mv.kind == "S1":
                                self.profile.S1_accepted += 1
                            elif mv.kind == "S2":
                                self.profile.S2_accepted += 1
                            self._register_best(st)
                            improved = True
                            break

            self._register_best(st)
            return st
        finally:
            self.profile.local_search_seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Elite pool
    # ------------------------------------------------------------------
    @staticmethod
    def hamming(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.count_nonzero(a != b))

    def _register_best(self, state: State) -> bool:
        if not self.feasible(state.y):
            return False
        improved = False
        if self.best_feasible is None or state.raw > self.best_feasible.raw + 1e-12:
            self.best_feasible = state.copy()
            improved = True
        return improved

    def _add_elite(self, state: State) -> None:
        if not self.feasible(state.y):
            return

        # If same solution already exists, keep best copy (objective should match).
        for q, e in enumerate(self.elite):
            if np.array_equal(e.y, state.y):
                if state.raw > e.raw:
                    self.elite[q] = state.copy()
                return

        # Diversity filter when pool is not empty.
        if self.elite:
            distances = [self.hamming(state.y, e.y) for e in self.elite]
            if min(distances) < self.cfg.elite_min_hamming:
                # Replace a too-similar inferior elite solution.
                q = int(np.argmin(distances))
                if state.raw > self.elite[q].raw + 1e-12:
                    self.elite[q] = state.copy()
                    self.elite.sort(key=lambda s: s.raw, reverse=True)
                return

        self.elite.append(state.copy())
        self.elite.sort(key=lambda s: s.raw, reverse=True)
        if len(self.elite) > self.cfg.elite_size:
            self.elite.pop()

    # ------------------------------------------------------------------
    # Adaptive perturbation
    # ------------------------------------------------------------------
    def _random_A_perturb(self, state: State) -> Optional[Move]:
        choices = []
        for j in range(self.n):
            mv = Move("A", remove=(j,)) if state.y[j] else Move("A", add=(j,))
            y2 = state.y.copy()
            y2[j] = 1 - y2[j]
            if self._controlled(y2):
                choices.append(mv)
        if not choices:
            return None
        return choices[int(self.rng.integers(len(choices)))]

    def _random_S1(self, state: State) -> Optional[Move]:
        feasible_groups = []
        for group in (self.K, self.J):
            open_idx = group[state.y[group] == 1]
            closed_idx = group[state.y[group] == 0]
            if len(open_idx) and len(closed_idx):
                feasible_groups.append((open_idx, closed_idx))
        if not feasible_groups:
            return None
        open_idx, closed_idx = feasible_groups[int(self.rng.integers(len(feasible_groups)))]
        return Move(
            "S1",
            remove=(int(self.rng.choice(open_idx)),),
            add=(int(self.rng.choice(closed_idx)),),
        )

    def perturb(
        self,
        state: State,
        strength: int,
        stagnation: int,
    ) -> State:
        self.profile.perturbation_calls += 1
        t0 = time.perf_counter()
        try:
            st = state.copy()

            extra = min(0.25, 0.02 * stagnation)
            base_weights = {
                "A": min(0.45, self.cfg.perturb_prob_A + extra),
                "S1": max(0.10, self.cfg.perturb_prob_S1 - 1.5 * extra),
                "S2": min(0.45, self.cfg.perturb_prob_S2 + 0.5 * extra),
            }
            enabled = []
            if self.cfg.enable_A:
                enabled.append("A")
            if self.cfg.enable_S1:
                enabled.append("S1")
            if self.cfg.enable_S2:
                enabled.append("S2")
            if not enabled:
                return st

            weights = np.asarray([max(_EPS, base_weights[k]) for k in enabled], dtype=float)
            weights /= weights.sum()

            for _ in range(strength):
                # Infeasibility can only be changed by A.
                if not self.feasible(st.y) and self.cfg.enable_A:
                    kind = "A"
                else:
                    kind = enabled[int(self.rng.choice(len(enabled), p=weights))]

                mv: Optional[Move]
                if kind == "A":
                    mv = self._random_A_perturb(st)
                elif kind == "S1":
                    mv = self._random_S1(st)
                else:
                    mv = self._sample_S2_move(st)

                # Fallback among enabled operators.
                if mv is None:
                    fallback = []
                    if self.cfg.enable_A:
                        fallback.append(self._random_A_perturb)
                    if self.cfg.enable_S1:
                        fallback.append(self._random_S1)
                    if self.cfg.enable_S2:
                        fallback.append(self._sample_S2_move)
                    self.rng.shuffle(fallback)
                    for fn in fallback:
                        mv = fn(st)
                        if mv is not None:
                            break
                if mv is None:
                    break

                raw, pen = self._move_value(st, mv)
                st = self._apply_move(
                    st,
                    mv,
                    raw=raw,
                    penalized=pen,
                    update_tabu=(mv.kind == "A"),
                )
                if mv.kind == "A":
                    self.profile.perturb_A_moves += 1
                elif mv.kind == "S1":
                    self.profile.perturb_S1_moves += 1
                elif mv.kind == "S2":
                    self.profile.perturb_S2_moves += 1

            return st
        finally:
            self.profile.perturbation_seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Path Relinking
    # ------------------------------------------------------------------
    def _target_directed_moves(
        self, state: State, guide: State
    ) -> list[Move]:
        """
        Moves that strictly reduce Hamming distance to guide.

        Priority is generated by S1/S2-style moves. Feasible single toggles
        are also included when elite solutions have different group counts.
        """
        y = state.y
        g = guide.y
        moves: list[Move] = []

        # S1 target-directed swaps within K or J.
        for group in (self.K, self.J):
            remove = group[(y[group] == 1) & (g[group] == 0)]
            add = group[(y[group] == 0) & (g[group] == 1)]
            for r in remove:
                for a in add:
                    moves.append(Move("PR-S1", remove=(int(r),), add=(int(a),)))

        # Sample target-directed S2 combinations.
        s1_moves = moves.copy()
        if len(s1_moves) >= 2:
            for _ in range(self.cfg.path_relink_s2_samples):
                q = self.rng.choice(len(s1_moves), size=2, replace=False)
                m1, m2 = s1_moves[int(q[0])], s1_moves[int(q[1])]
                rem = tuple(dict.fromkeys(m1.remove + m2.remove))
                add = tuple(dict.fromkeys(m1.add + m2.add))
                if len(rem) == 2 and len(add) == 2 and not set(rem) & set(add):
                    moves.append(Move("PR-S2", remove=rem, add=add))

        # A-style target-directed toggles, but PR itself remains feasible.
        diff = np.flatnonzero(y != g)
        for j in diff:
            if y[j] == 1:
                mv = Move("PR-A", remove=(int(j),))
            else:
                mv = Move("PR-A", add=(int(j),))
            y2 = y.copy()
            y2[j] = g[j]
            if self.feasible(y2):
                moves.append(mv)

        # Keep only moves that are feasible and reduce Hamming distance.
        d0 = self.hamming(y, g)
        unique = {}
        for mv in moves:
            y2 = y.copy()
            if mv.remove:
                y2[list(mv.remove)] = 0
            if mv.add:
                y2[list(mv.add)] = 1
            if self.feasible(y2) and self.hamming(y2, g) < d0:
                key = (tuple(sorted(mv.remove)), tuple(sorted(mv.add)))
                unique[key] = mv
        return list(unique.values())

    def _path_relink_one_way(self, start: State, guide: State) -> State:
        st = start.copy()
        best = st.copy()

        while not np.array_equal(st.y, guide.y):
            moves = self._target_directed_moves(st, guide)
            if not moves:
                break

            best_mv = None
            best_raw = -math.inf
            best_pen = -math.inf
            for mv in moves:
                raw, pen = self._move_value(st, mv)
                self.profile.PR_move_evaluations += 1
                if raw > best_raw + 1e-12:
                    best_mv, best_raw, best_pen = mv, raw, pen

            if best_mv is None:
                break
            st = self._apply_move(st, best_mv, raw=best_raw, penalized=best_pen)
            self.profile.PR_steps += 1
            if best_mv.kind == "PR-A":
                self.profile.PR_A_steps += 1
            elif best_mv.kind == "PR-S1":
                self.profile.PR_S1_steps += 1
            elif best_mv.kind == "PR-S2":
                self.profile.PR_S2_steps += 1

            if st.raw > best.raw + 1e-12:
                best = st.copy()
            self._register_best(st)

        return best

    def path_relink(self, a: State, b: State) -> State:
        """
        Perform PR and then VND intensification.

        Profiling deliberately times the PR core separately; the final VND is
        recorded under local_search_seconds so component times do not overlap.
        """
        self.profile.path_relink_calls += 1
        t0 = time.perf_counter()
        c1 = self._path_relink_one_way(a, b)
        if self.cfg.path_relink_bidirectional:
            c2 = self._path_relink_one_way(b, a)
            candidate = c2 if c2.raw > c1.raw else c1
        else:
            candidate = c1
        self.profile.path_relink_seconds += time.perf_counter() - t0
        return self.local_search(candidate)

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------
    def _scale_penalty_if_needed(self, reference: State) -> None:
        if self.rho is not None:
            return
        # One unit of violation should be material relative to the full MNL
        # objective, but not so huge that controlled infeasible exploration
        # becomes impossible.
        scale = max(1.0, abs(reference.raw))
        denom = max(1, self.p)
        self.rho = self.cfg.penalty_multiplier * scale / denom
        reference.penalized = self._penalized(reference.raw, reference.y)

    def solve(self) -> SearchResult:
        start_time = time.perf_counter()
        self.profile = SearchProfile()
        self.best_feasible = None
        self.elite = []
        self.history = []
        self.move_clock = 0
        self.tabu_drop_until.fill(-1)
        self.tabu_add_until.fill(-1)

        # ---------------- GRASP multi-start + local improvement ----------------
        current: Optional[State] = None
        for s in range(self.cfg.grasp_starts):
            st = self.grasp_construct()
            if self.rho is None:
                self._scale_penalty_if_needed(st)
                st.penalized = self._penalized(st.raw, st.y)

            self._register_best(st)
            st = self.local_search(st)
            self._add_elite(st)
            self._register_best(st)

            if current is None or st.raw > current.raw:
                current = st.copy()

            if self._time_exceeded(start_time):
                break

        if current is None or self.best_feasible is None:
            raise RuntimeError("No feasible solution was generated.")

        # ---------------- Iterated Local Search ----------------
        strength = int(
            np.clip(
                self.cfg.initial_perturbation_strength,
                self.cfg.min_perturbation_strength,
                self.cfg.max_perturbation_strength,
            )
        )
        stagnation = 0
        iterations_done = 0

        for it in range(1, self.cfg.max_iterations + 1):
            if self._time_exceeded(start_time):
                break
            iterations_done = it
            before_best = self.best_feasible.raw

            shaken = self.perturb(current, strength, stagnation)
            candidate = self.local_search(shaken)
            self._add_elite(candidate)
            improved_global = self._register_best(candidate)

            # Record-to-record acceptance around the best feasible value.
            threshold = self.best_feasible.raw - self.cfg.acceptance_tolerance * (
                1.0 + abs(self.best_feasible.raw)
            )
            if candidate.raw >= current.raw - 1e-12 or candidate.raw >= threshold:
                current = candidate.copy()
            elif self.elite:
                # Diversification restart from a random elite solution.
                current = self.elite[int(self.rng.integers(len(self.elite)))].copy()

            # Periodic Path Relinking with a diverse elite guide.
            if (
                self.cfg.enable_path_relink
                and self.cfg.path_relink_interval > 0
                and it % self.cfg.path_relink_interval == 0
                and len(self.elite) >= 2
            ):
                distances = np.asarray(
                    [self.hamming(current.y, e.y) for e in self.elite], dtype=int
                )
                # Pick among the most distant half to favor diversity.
                order = np.argsort(-distances)
                half = order[: max(1, len(order) // 2)]
                guide_idx = int(self.rng.choice(half))
                guide = self.elite[guide_idx]

                if not np.array_equal(current.y, guide.y):
                    pr = self.path_relink(current, guide)
                    self._add_elite(pr)
                    self._register_best(pr)
                    if pr.raw > current.raw + 1e-12:
                        current = pr.copy()

            after_best = self.best_feasible.raw
            if after_best > before_best + 1e-12 or improved_global:
                stagnation = 0
                strength = max(
                    self.cfg.min_perturbation_strength,
                    strength - 1,
                )
            else:
                stagnation += 1
                if stagnation % self.cfg.stagnation_window == 0:
                    strength = min(
                        self.cfg.max_perturbation_strength,
                        strength + 1,
                    )

            self.history.append(
                {
                    "iteration": it,
                    "best": self.best_feasible.raw,
                    "current": current.raw,
                    "strength": strength,
                    "stagnation": stagnation,
                    "elite_size": len(self.elite),
                }
            )

            if self.cfg.verbose and (
                it == 1 or it % 25 == 0 or after_best > before_best + 1e-12
            ):
                print(
                    f"[ILS {it:4d}] best={self.best_feasible.raw:.8f} "
                    f"current={current.raw:.8f} strength={strength} "
                    f"elite={len(self.elite)}"
                )

        best = self.best_feasible
        elapsed = time.perf_counter() - start_time
        self.profile.total_seconds = elapsed
        open_existing = np.flatnonzero(best.y[: self.n_k] == 1).astype(int).tolist()
        closed_existing = np.flatnonzero(best.y[: self.n_k] == 0).astype(int).tolist()
        open_candidates = (
            np.flatnonzero(best.y[self.n_k :] == 1).astype(int).tolist()
        )

        return SearchResult(
            objective=float(best.raw),
            penalized_objective=float(best.penalized),
            feasible=self.feasible(best.y),
            y=best.y.copy(),
            open_existing=open_existing,
            closed_existing=closed_existing,
            open_candidates=open_candidates,
            iterations=iterations_done,
            elapsed_seconds=elapsed,
            perturbation_strength=strength,
            profile=self.profile.copy(),
            history=self.history.copy(),
        )

    def _time_exceeded(self, start_time: float) -> bool:
        return (
            self.cfg.time_limit is not None
            and (time.perf_counter() - start_time) >= self.cfg.time_limit
        )


def run_cdata_example(
    N: int = 100,
    p: int = 10,
    lambda_: float = 1.0,
    instance_seed: int = 1,
    search_seed: int = 123,
    iterations: int = 200,
    time_limit: Optional[float] = None,
    verbose: bool = True,
) -> SearchResult:
    """
    Create a and b through the user's CData generator and solve the instance.
    """
    # Delayed import keeps this heuristic module testable independently.
    from data import CData

    instance = CData.from_total_nodes(
        N=N,
        p=p,
        lambda_=lambda_,
        seed=instance_seed,
    )

    cfg = ILSConfig(
        seed=search_seed,
        max_iterations=iterations,
        time_limit=time_limit,
        verbose=verbose,
    )
    solver = MNLParcelLockerILS(instance, cfg)
    return solver.solve()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ILS/GRASP/VND/Path-Relinking heuristic for MNL locker location."
    )
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--p", type=int, default=10)
    p.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    p.add_argument("--instance-seed", type=int, default=1)
    p.add_argument("--search-seed", type=int, default=123)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = run_cdata_example(
        N=args.N,
        p=args.p,
        lambda_=args.lambda_,
        instance_seed=args.instance_seed,
        search_seed=args.search_seed,
        iterations=args.iterations,
        time_limit=args.time_limit,
        verbose=not args.quiet,
    )
    print()
    print(result)
    print("Open candidate lockers:", result.open_candidates)
    print("Closed existing POP stations:", result.closed_existing)
    print()
    print(result.profile_report())
