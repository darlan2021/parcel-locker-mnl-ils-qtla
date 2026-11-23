from __future__ import annotations

from mthds.dinkelbach import get_dinkelbach_solution
from pathlib import Path
from typing import Sequence
import json

import numpy as np

GRIDSIZE = 1000.0
DISTANCE_SCALE = 100.0
LIN_ALPHA_VALUES = (0.5, 1.0, 2.0)

class CData:
    """
    Lin et al.-compatible synthetic instance generator for the
    pop-station / candidate-locker network redesign problem.

    sets
    ----
    I : customer zones, size n_i
    J : candidate locker locations, size n_j
    K : existing POP-stations, size n_k

    Exact P-for-P replacement version
    ---------------------------------
        sum_{j in J} x_j = p
        sum_{k in K} r_k = n_k - p

    where x_j = 1 means candidate locker j is opened and
    r_k = 1 means existing POP-station k remains open.

    synthetic-data specification follows Lin et al. (2020):
      * coordinates for I, J, K ~ Uniform([0,1000]^2)
      * demand d_i ~ Uniform([0,100])
      * Euclidean distance measured in units of 100
      * v_im = -alpha * L_im
      * b_im = exp(v_im)
      * service score:
            1.0  if L <= 1
            0.5  if 1 < L <= 2
            0.2  if 2 < L <= 3
            0.0  if L > 3

    the canonical matrices b_existing and b_candidate are the raw MNL
    attractiveness weights. No epsilon and no clipping are applied.

    the matrices b and a concatanate the matrices b_existing and b_candidate, and
    a_existing and a_candidate, respectively.
    """
    def __init__(
        self,
        n_i: int,
        n_j: int | None,
        p: int,
        lambda_: float,
        seed: int,
        n_k: int | None = None,
        *,
        grid_size: float = GRIDSIZE,
        distance_scale: float = DISTANCE_SCALE,
        denominator_warning: float = 1e-12,
    ):
        n_i = int(n_i)
        p = int(p)
        seed = int(seed)
        lambda_ = float(lambda_)
        # check input validity
        if n_i <= 0:
            raise ValueError("n_i must be positive.")

        if n_j is None:
            if n_i % 2 != 0:
                raise ValueError(
                    "n_i must be even when n_j is omitted, because "
                    "Lin-style generation sets n_j = n_i / 2."
                )
            n_j = n_i // 2
        n_j = int(n_j)

        if n_k is None:
            n_k = n_j
        n_k = int(n_k)

        if n_j <= 0 or n_k <= 0:
            raise ValueError("n_j and n_k must be positive.")
        if not 1 <= p <= n_j:
            raise ValueError(f"p must satisfy 1 <= p <= n_j={n_j}.")
        if p > n_k:
            raise ValueError(
                f"For P-for-P replacement, p must satisfy p <= n_k={n_k}."
            )
        if lambda_ <= 0.0:
            raise ValueError("lambda_ (MNL distance sensitivity) must be positive.")
        if grid_size <= 0.0 or distance_scale <= 0.0:
            raise ValueError("grid_size and distance_scale must be positive.")

        self.n_i = n_i
        self.n_j = n_j
        self.n_k = n_k
        self.ni = n_i
        self.nj = n_j
        self.nk = n_k
        self.tn = n_i + n_k + n_j
        self.n = n_k + n_j
        self.p = p
        self.q = n_k - p
        self.lambda_ = lambda_
        self.alpha = lambda_
        self.seed = seed
        self.grid_size = float(grid_size)
        self.distance_scale = float(distance_scale)
        self.denominator_warning = float(denominator_warning)
        self.N = range(self.nj + self.nk)
        self.J = range(self.nj)
        self.I = range(self.ni)
        self.K = range(self.nk)

        # coordinates for I, K, J.
        rng = np.random.default_rng(self.seed)
        self.customer_xy = rng.uniform(0.0, self.grid_size, size=(self.n_i, 2))
        self.existing_xy = rng.uniform(0.0, self.grid_size, size=(self.n_k, 2))
        self.candidate_xy = rng.uniform(0.0, self.grid_size, size=(self.n_j, 2))

        # Lin artificial instances: demand ~ Uniform[0,100].
        self.demand = rng.uniform(1, 100, size=self.n_i).astype(int)

        # distances.
        self.distance_existing = self._distances(self.customer_xy, self.existing_xy)
        self.distance_candidate = self._distances(self.customer_xy, self.candidate_xy)

        # Lin-style service score.
        self.service_thresholds = np.asarray((1.0, 2.0, 3.0), dtype=float)
        self.service_levels = np.asarray((1.0, 0.5, 0.2, 0.0), dtype=float)
        self.service_existing = self._service_matrix(self.distance_existing)
        self.service_candidate = self._service_matrix(self.distance_candidate)

        # v_im = -alpha L_im; b_im = exp(v_im).
        self.log_b_existing = -self.lambda_ * self.distance_existing
        self.log_b_candidate = -self.lambda_ * self.distance_candidate
        self.b_existing = np.exp(self.log_b_existing)
        self.b_candidate = np.exp(self.log_b_candidate)

        if not np.all(np.isfinite(self.b_existing)):
            raise FloatingPointError("Non-finite existing-facility MNL coefficients generated.")
        if not np.all(np.isfinite(self.b_candidate)):
            raise FloatingPointError("Non-finite candidate-locker MNL coefficients generated.")
        if np.any(self.b_existing <= 0.0) or np.any(self.b_candidate <= 0.0):
            raise FloatingPointError(
                "A non-positive MNL coefficient was generated. "
                "With Lin's parameter ranges this should not occur."
            )

        # a_im = d_i * service_im * b_im.
        self.a_existing = self.demand[:, None] * self.service_existing * self.b_existing
        self.a_candidate = self.demand[:, None] * self.service_candidate * self.b_candidate

        # short aliases useful in mathematical-programming code.
        self.bk = self.b_existing
        self.bj = self.b_candidate
        self.ak = self.a_existing
        self.aj = self.a_candidate

        # concatenate to help in mathematical-programming code
        self.b = np.hstack((self.bk,self.bj))
        self.a = np.hstack((self.ak,self.aj))
        
        # naming instance
        alpha_tag = str(self.lambda_).replace(".", "p")
        self.name = (
            f"N{self.tn}_I{self.n_i}_K{self.n_k}_J{self.n_j}_"
            f"P{self.p}_a{alpha_tag}_s{self.seed}"
        )

        # eta initial upper bound
        self.U = np.zeros(self.ni,dtype=float)
        for i in self.I:
          self.U[i], _ = get_dinkelbach_solution(self.a[i], self.b[i], self.p)
        self.U_sum = np.sum(self.U)

        self.metadata = {
            "generator": "CData-Lin-compatible",
            "problem": "P-for-P POP-station / locker network redesign",
            "N": self.n,
            "n_i": self.n_i,
            "n_j": self.n_j,
            "n_k": self.n_k,
            "p": self.p,
            "alpha": self.lambda_,
            "seed": self.seed,
            "grid_size": self.grid_size,
            "distance_scale": self.distance_scale,
            "demand_distribution": "Uniform[0,100]",
            "coordinate_distribution": f"Uniform([0,{self.grid_size}]^2)",
            "service_thresholds": self.service_thresholds.tolist(),
            "service_levels": self.service_levels.tolist(),
        }

    @classmethod
    def from_total_nodes(
        cls,
        N: int,
        p: int,
        lambda_: float,
        seed: int,
        **kwargs,
    ) -> "CData":
        """
        Lin artificial-network proportions:
            n_i = N/2, n_k = N/4, n_j = N/4.

        Examples:
            N=60  -> I=30, K=15, J=15
            N=80  -> I=40, K=20, J=20
            N=100 -> I=50, K=25, J=25
            N=400 -> I=200, K=100, J=100
        """
        N = int(N)
        if N % 4 != 0:
            raise ValueError(
                "N must be divisible by 4 for Lin-style proportions "
                "(I=N/2, K=N/4, J=N/4)."
            )
        return cls(
            n_i=N // 2,
            n_j=N // 4,
            n_k=N // 4,
            p=p,
            lambda_=lambda_,
            seed=seed,
            **kwargs,
        )

    def _distances(
        self,
        customer_xy: np.ndarray,
        facility_xy: np.ndarray,
    ) -> np.ndarray:
        delta = customer_xy[:, None, :] - facility_xy[None, :, :]
        raw_distance = np.sqrt(np.sum(delta * delta, axis=2))
        return raw_distance / self.distance_scale

    def _service_matrix(self, distance: np.ndarray) -> np.ndarray:
        bins = np.digitize(distance, bins=self.service_thresholds, right=True)
        return self.service_levels[bins]

    def solution_from_sets(
        self,
        open_candidates: Sequence[int],
        closed_existing: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a feasible (x,r) pair."""
        open_candidates = np.asarray(open_candidates, dtype=int)
        closed_existing = np.asarray(closed_existing, dtype=int)

        if len(open_candidates) != self.p:
            raise ValueError(
                f"Expected p={self.p} open candidate lockers, "
                f"received {len(open_candidates)}."
            )
        if len(closed_existing) != self.p:
            raise ValueError(
                f"Expected p={self.p} closed existing POP stations, "
                f"received {len(closed_existing)}."
            )
        if len(np.unique(open_candidates)) != len(open_candidates):
            raise ValueError("Duplicate candidate-locker indices are not allowed.")
        if len(np.unique(closed_existing)) != len(closed_existing):
            raise ValueError("Duplicate existing-station indices are not allowed.")
        if np.any(open_candidates < 0) or np.any(open_candidates >= self.n_j):
            raise ValueError("Candidate-locker index outside valid range.")
        if np.any(closed_existing < 0) or np.any(closed_existing >= self.n_k):
            raise ValueError("Existing-station index outside valid range.")

        x = np.zeros(self.n_j, dtype=np.int8)
        x[open_candidates] = 1
        r = np.ones(self.n_k, dtype=np.int8)
        r[closed_existing] = 0
        self.validate_solution(x, r)
        return x, r

    def save(self, directory: str | Path) -> None:
        """Save raw Lin-compatible instance data and diagnostics."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            directory / f"{self.name}.npz",
            customer_xy=self.customer_xy,
            existing_xy=self.existing_xy,
            candidate_xy=self.candidate_xy,
            demand=self.demand,
            distance_existing=self.distance_existing,
            distance_candidate=self.distance_candidate,
            service_existing=self.service_existing,
            service_candidate=self.service_candidate,
            log_b_existing=self.log_b_existing,
            log_b_candidate=self.log_b_candidate,
            b_existing=self.b_existing,
            b_candidate=self.b_candidate,
            a_existing=self.a_existing,
            a_candidate=self.a_candidate,
            B_min=self.B_min,
            B_max=self.B_max,
            z_lower=self.z_lower,
            z_upper=self.z_upper,
        )

        with open(directory / f"{self.name}.json", "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2)

