import numpy as np
import gurobipy as gp
from gurobipy import GRB

rng = np.random.default_rng()

def match(S, M):
    # loads of 3, 3
    revloads = np.full(S.shape[0], 3)
    paploads = np.full(S.shape[1], 3)

    model = gp.Model("my_model") 
    model.setParam('OutputFlag', 0)
    obj = 0
    n, d = S.shape
    assert len(revloads) == n and len(paploads) == d and S.shape == M.shape
    review_demand = np.sum(paploads)
    review_supply = np.sum(revloads)

    if review_demand > review_supply:
        print('infeasible assignment')
        raise RuntimeError('infeasible')

    A = [[None for j in range(d)] for i in range(n)]

    for i in range(n):
        for j in range(d):
            if (M[i, j] == 1):
                v = model.addVar(lb = 0, ub = 0, name = f"{i} {j}")
            else:
                v = model.addVar(lb = 0, ub = 1, name = f"{i} {j}") 
                
            A[i][j] = v
            obj += v * S[i, j]

    model.setObjective(obj, GRB.MAXIMIZE)
    
    for i in range(n):
        papers = 0
        for j in range(d):
            papers += A[i][j]
        model.addConstr(papers <= revloads[i]) 
    
    for j in range(d):
        reviewers = 0
        for i in range(n):
            reviewers += A[i][j]
        model.addConstr(reviewers == paploads[j])
    
    model.optimize()
    #print(model.objVal)

    if model.status != GRB.OPTIMAL:
        print("WARNING: model not solved")
        raise RuntimeError('unsolved')

    B = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            B[i, j] = A[i][j].x

    return B 
