from data import CData
from mthds.mip import mip
from mnl_ils import ILSConfig, MNLParcelLockerILS
import sys
import numpy as np

import time


def main():
    n = 100
    p = 3
    seed = 1
    lambda_ = 0.5
    #lambda_ = 1
    #lambda_ = 2

    dt = CData.from_total_nodes(N=n,p=p,lambda_=lambda_,seed=seed)
    config = ILSConfig(
                grasp_starts=8,
                #max_iterations=300,
                max_iterations=10,
                tau_tabu=7,
                max_cardinality_violation=2,
                initial_perturbation_strength=2,
                max_perturbation_strength=10,
                elite_size=8,
                path_relink_interval=10,
                seed=123,
            )

    solver = MNLParcelLockerILS(dt, config)
    result = solver.solve()
 
    print("Objective:", result.objective)
    print("Candidate lockers opened:", result.open_candidates)
    print("Existing POP stations closed:", result.closed_existing)
    print("Running time:", result.elapsed_seconds, "seconds")
    print(result.profile_report())
    print()
    print()
    mip(dt)
    print()
    print()
if __name__ == "__main__":
   main() 
    
