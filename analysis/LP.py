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

def match(S, M):
    # loads of 3, 3
    #####
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)

    assign_vars = {}
    obj = 0

    stime = time.time()
    print('Constructing LP')
    nrev, npap = S.shape
    assert nrev == npap
    for r, p in product(range(nrev), range(npap)):
        if M[r, p] == 1:
            ub = 0
        else:
            ub = 1 
        v = m.addVar(lb=0, ub=ub, name=f'{r},{p}')
        obj += v * S[r, p]
        assign_vars[r, p] = v
    m.setObjective(obj, GRB.MAXIMIZE)

    for p in range(npap):
        m.addConstr(gp.quicksum(assign_vars[r, p] for r in range(nrev)) == 3)
    for r in range(nrev):
        m.addConstr(gp.quicksum(assign_vars[r, p] for p in range(npap)) == 3)
    print('Done constructing', time.time() - stime)
    stime = time.time()

    print('Solving LP')
    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')
    print('Done solving', time.time() - stime)
    stime = time.time()

    print('Outputting LP')    
    F = np.zeros((nrev, npap))
    for r, p in product(range(nrev), range(npap)):
        F[r, p] = assign_vars[r, p].x
    print('Done outputting', time.time() - stime)
    return F

