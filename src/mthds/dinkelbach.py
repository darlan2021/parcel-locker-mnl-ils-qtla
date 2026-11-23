import numpy as np

def get_dinkelbach_solution(a, b, p, tol=1e-9, max_iter=100):
    """
    Solves max <a,x>/<b,x> subject to ||x||_1 = p, x binary, a,b >= 0
    using an O(n) per-iteration Dinkelbach approach.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    
    if p <= 0 or p > n:
        raise ValueError("p must be between 1 and the length of the input vectors.")

    # initial a valid subset (first p elements)
    # a zero denominator is avoided 
    sum_b = np.sum(b[:p])
    if sum_b > 0:
        lmbda = np.sum(a[:p]) / sum_b
    else:
        # otherwise : pick elements with the largest b values to avoid division by zero
        initial_idx = np.argpartition(-b, p - 1)[:p]
        lmbda = np.sum(a[initial_idx]) / np.sum(b[initial_idx])
        
    for _ in range(max_iter):
        c = a - lmbda * b
        
        # find indices of the p largest elements 
        # np.argpartition puts the smallest p elements of -c at the beginning
        idx = np.argpartition(-c, p - 1)[:p]
        
        # f(lambda) = max <a - lambda*b, x>
        sum_a_next = np.sum(a[idx])
        sum_b_next = np.sum(b[idx])
        f_val = sum_a_next - lmbda * sum_b_next
        
        # check for convergence
        if f_val < tol:
            break
            
        # update lambda
        lmbda = sum_a_next / sum_b_next
        
    # the binary solution vector
    x = np.zeros(n, dtype=int)
    x[idx] = 1
    return lmbda, x

def constrained_dinkelbach(a, b, p, x_partial, tol=1e-9, max_iter=100):
    """
    solves max <a,x>/<b,x> subject to ||x||_1 = p, x binary, a,b >= 0
    given a partial solution array (1: fixed in, 0: fixed out, -1: free).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x_partial = np.asarray(x_partial, dtype=int)
    n = len(a)
    
    fixed_in = (x_partial == 1)
    free = (x_partial == -1)
    
    num_fixed_in = np.sum(fixed_in)
    num_free = np.sum(free)
    p_prime = p - num_fixed_in
    
    if p_prime < 0 or p_prime > num_free:
        raise ValueError("infeasible: fixed facilities exceed p, or not enough free candidates to reach p.")
        
    # precompute a and bf of fixed facilities 
    A_fixed = np.sum(a[fixed_in])
    B_fixed = np.sum(b[fixed_in])
    
    # isolate free variables
    a_free = a[free]
    b_free = b[free]
    free_indices = np.where(free)[0] # map back to original array indices
    
    # initialize lambda safely
    if p_prime == 0:
        # all required elements are already fixed; no choices left to make.
        return A_fixed / B_fixed, np.where(x_partial == 1, 1, 0)
        
    sum_b_init = B_fixed + np.sum(b_free[:p_prime])
    if sum_b_init > 0:
        lmbda = (A_fixed + np.sum(a_free[:p_prime])) / sum_b_init
    else:
        init_idx = np.argpartition(-b_free, p_prime - 1)[:p_prime]
        lmbda = (A_fixed + np.sum(a_free[init_idx])) / (B_fixed + np.sum(b_free[init_idx]))

    # dinkelbach main loop
    for _ in range(max_iter):
        # compute weights for free variables only
        c_free = a_free - lmbda * b_free
        
        # select top p' indices from the free variables
        local_idx = np.argpartition(-c_free, p_prime - 1)[:p_prime]
        
        # xalculate new sums including fixed parts
        sum_a_next = A_fixed + np.sum(a_free[local_idx])
        sum_b_next = B_fixed + np.sum(b_free[local_idx])
        
        # evaluate f(lambda)
        f_val = sum_a_next - lmbda * sum_b_next
        
        if f_val < tol:
            break
            
        lmbda = sum_a_next / sum_b_next
        
    # construct final solution vector
    x_final = np.zeros(n, dtype=int)
    x_final[fixed_in] = 1
    
    # map the chosen local free indices back to global positions
    global_chosen_idx = free_indices[local_idx]
    x_final[global_chosen_idx] = 1
    
    return lmbda, x_final