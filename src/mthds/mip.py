import gurobipy as gp
from gurobipy import GRB
from data import CData
import numpy as np
from time import perf_counter as timer

GUROBI_STATUS_MSG ={
   1 :  "LOADED",   
   2 :  "OPTIMAL",    
   3 :  "INFEASIBLE",    
   4 :  "INF_OR_UNBD",    
   5 :  "UNBOUNDED",    
   6 :  "CUTOFF",    
   7 :  "ITERATION_LIMIT",
   8 :  "NODE_LIMIT",    
   9 :  "TIME_LIMIT",    
  10 :  "SOLUTION_LIMIT",    
  11 :  "INTERRUPTED",    
  12 :  "NUMERIC",    
  13 :  "SUBOPTIMAL",    
  14 :  "INPROGRESS",    
  15 :  "USER_OBJ_LIMIT",    
  16 :  "WORK_LIMIT",    
  17 :  "MEM_LIMIT"
}

def mip(dt):
    U_x1,L_x0,L_x1 = np.zeros((dt.ni,dt.n)),np.zeros((dt.ni,dt.n)),np.zeros((dt.ni,dt.n))
    U = np.zeros(dt.ni)

    sum_bk = np.sum(dt.b[:,:dt.nk],axis=1)
    for i in dt.I:
        bk_i = dt.b[i,:dt.nk]
        sorted_bk_i = np.sort(bk_i)

        bj_i = dt.b[i,dt.nk:]
        sorted_bj_i_desc = np.sort(bj_i)[::-1]

        sum_bk_smallest_nk_p = np.sum(sorted_bk_i[:dt.nk-dt.p])
        U[i] =  1.0/sum_bk_smallest_nk_p

        sum_bj_largest_p = np.sum(sorted_bj_i_desc[:dt.p])
        for j in dt.N:
            if j < dt.nk:
               # Extract elements in K \ {k} and sort ascending
               bk_i_without_k_sorted = np.sort(np.delete(bk_i, j))
               
               # Equation (25)
               U_x1[i, j] = 1.0 / (np.sum(bk_i_without_k_sorted[:dt.nk - dt.p - 1]) + bk_i[j])
               
               # Equation (26)
               sum_bk_without_k = sum_bk[i] - bk_i[j]
               L_x0[i, j] = 1.0 / (sum_bk_without_k + sum_bj_largest_p )
               
               # Equation (27)
               L_x1[i, j] = 1.0 / (sum_bk[i] + sum_bj_largest_p)
            else:
               bj_i_without_j_desc = np.sort(np.delete(bj_i, j-dt.nk))[::-1]
               # Equation (22)
               U_x1[i, j] = 1.0 / (sum_bk_smallest_nk_p + bj_i[j-dt.nk])
               # Equation (23)
               L_x0[i, j] = 1.0 / (sum_bk[i] + np.sum(bj_i_without_j_desc[:dt.p]))
               # Equation (24)
               if dt.p > 1:
                   L_x1[i, j] = 1.0 / (sum_bk[i] + bj_i[j-dt.nk] + np.sum(bj_i_without_j_desc[:dt.p - 1]))
               else:
                   L_x1[i, j] = 1.0 / (sum_bk[i] + bj_i[j-dt.nk]) 

    model = gp.Model("Sum_of_Ratios_Linearized")
    model.setParam('Threads', 1)
    model.setParam("OutputFlag", 1)

    x = model.addVars(dt.N, vtype=GRB.BINARY, name="x")
    z = model.addVars(dt.I, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
    y = model.addVars(dt.I, dt.N, vtype=GRB.CONTINUOUS, lb=0.0, name="y")

    model.setObjective(
        gp.quicksum(dt.a[i][j] * y[i, j] for i in dt.I for j in dt.N), 
        GRB.MAXIMIZE
    )

    model.addConstr(gp.quicksum(x[dt.nk+j] for j in dt.J) <= dt.p)
    model.addConstr(gp.quicksum(x[k] for k in dt.K) >= dt.nk - dt.p)

    for i in dt.I:
        model.addConstr(gp.quicksum(dt.b[i][k] * y[i, k] for k in dt.N) == 1)

        for k in dt.N:
            #(28) and (30)
            model.addConstr(y[i, k] >= z[i] - U[i] * (1 - x[k]))
            model.addConstr(y[i, k] <= z[i] - L_x0[i,k] * (1 - x[k]))
            #(29) and (31)
            model.addConstr(y[i, k] >= L_x1[i,k] * x[k])
            model.addConstr(y[i, k] <= U_x1[i,k] * x[k])

    starting_time = timer()
    model.optimize()
    cpu_time = timer() - starting_time

    # Display results
    if model.status == GRB.OPTIMAL:
        x_val = np.array([1 if x[j].X > 0.5 else 0 for j in dt.N])
        print(f"\n{'-'*40} {'MIP ':5s} {'-'*40}")
        print(f"instance         : (ni,nj,p,alpha,zero) = ({dt.ni},{dt.nj},{dt.p},{dt.alpha})")
        print(f"Status           : {GUROBI_STATUS_MSG[model.Status]:>18s}")
        print(f"Obj value        : {model.ObjVal:18.6f}")
        
        num_val = dt.a @ x_val
        den_val = dt.b @ x_val
        real_sum = (num_val/den_val).sum()

        sopen_facilities = ' '.join(map(str,[j for j in dt.N if x[j].X > 0.5]))
        
        print(f"Real obj value   : {real_sum:18.6f}")
        print(f"cpu time         : {cpu_time:18.2f}s")
        print(f"#bb              : {int(model.NodeCount):18d}")
        print(f"Opened facilities: {sopen_facilities:>18s}")