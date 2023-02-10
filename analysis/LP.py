import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from itertools import product

def fast_match(S, M): # r, p
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)
    F = m.addMVar(S.shape, lb=0, ub=(1-M), obj=S)
    m.modelSense = gp.GRB.MAXIMIZE
    for r in range(S.shape[0]):
        m.addConstr(F[r, :] @ np.ones(S.shape[1]) == 3)
    for p in range(S.shape[1]):
        m.addConstr(np.ones(S.shape[0]) @ F[:, p] <= 3)

    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    F_ = F.x
    return F_
