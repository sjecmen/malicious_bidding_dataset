import numpy as np
from utils import make_SA_matrix

# Define functions that bid for each reviewer type
# Each reviewer type applies to the whole group
# how to find X and Y? look at distribution and sample, and add noise(?)

# TODO steps still
# load all data from other script
# run experiments

# don't scale up number of positive bids for anyone, since these are relevant to load
# scale up negative bids since these aren't
rng = np.random.default_rng(0)

def copy_bid_distribution(B_row, SA_row, targets=[]):
    def random_bid_subset(B, papers, npos, nneg):
        nneg = min(nneg, papers - npos) # clip neg if too many
        sequence = rng.permutation(papers)
        B[sequence[:npos]] = 1
        B[sequence[npos:npos+nneg]] = -1

    npap0 = B_row.size
    npap1 = SA_row.size

    papers_sim = {x[0] for x in np.argwhere(SA_row > 0)} - set(targets)
    papers_non = {x[0] for x in np.argwhere(SA_row == 0)} - set(targets)

    # npos0/nneg0 does not count any bids on targets
    pos = B_row == 1
    neg = B_row == -1
    npos0 = [np.sum(pos[papers_sim]), np.sum(pos[papers_non])]
    nneg0 = [np.sum(neg[papers_sim]), np.sum(neg[papers_non])]
    npos1 = npos0 # array len 2
    nneg1 = [np.round(x * (npap1 / npap0)) for x in nneg0] # array len2

    B_new = np.zeros(npap1)
    random_bid_subset(B_new, papers_sim, npos1[0], nneg1[0])
    random_bid_subset(B_new, papers_non, npos1[1], nneg1[1])
    return B_new

def copy_bids(nrev, B_old, SA_new, models, targets=[]):
    assert len(targets) == 0 or len(targets) == nrev
    B_new = np.zeros(nrev, SA_new.shape[1]))
    for rid in range(nrev):
        model_rev = rng.choice(models)
        B_row = B_old[model_rev, :]
        SA_row = SA_new[rid, :]
        B_new[rid, :] = copy_bid_distribution(B_row, SA_row, targets)
    return B_new

def construct_SA_new(nrev1, npap1, group_size, reviewer_to_sas, paper_to_sa, group_map, reviewer_to_authored_sa):
    new_paper_to_sas = {}
    new_sa_to_papers = {}
    for pid in range(npap1):
        sa = rng.choice(paper_to_sas.values())
        new_paper_to_sas[pid] = sa
        if sa not in new_sa_to_papers:
            new_sa_to_papers[sa] = {pid}
        else:
            new_sa_to_papers[sa].add(pid)

    groups_of_size = {x for _, x in group_map.items() if len(x) == group_size}
    model_group = rng.choice(groups_of_size)
    new_reviewer_to_sas = {}
    targets = [] 
    M = np.zeros((nrev1, npap1)) # no COIs for honest revs
    for rid, model in enumerate(model_group):
        new_reviewer_to_sas[rid] = reviewer_to_sas[model]
        target_sa = reviewer_to_authored_sa[model]
        target = rng.choice(new_sa_to_papers[target_sa] - set(targets))
        targets.append(target)
        M[rid, target] = 1

    for rid in range(group_size, nrev1):
        sas = rng.choice(reviewer_to_sas.values())
        new_reviewer_to_sas[rid] = sas
    
    SA = make_SA_matrix(nrev1, npap1, new_reviewer_to_sas, new_sa_to_papers)
    return SA, M, list(range(group_size)), targets

def construct_honest_bid_matrix(B_old_honest, SA_new):
    nrev0, npap0 = B_old_honest.shape
    nrev1, npap1 = SA_new.shape
    B_new = copy_bids(nrev1, B_old_honest, set(range(nrev0)))
    return B_new

# eager to mixed set of target and non-group papers
# SA_new is truncated to len targets
def bid_malicious__basic(B_old_malicious, SA_new, targets):
    # for each in malicious, random sample a basic reviewer from B_old and copy
    assert len(targets) == SA_new.shape[0]
    B_new = copy_bids(len(targets), B_old_malicious, SA_new, strategy_map[0], targets)
    B_new[:, targets] = 1
    return B_new

# eager/indifferent to shared set of non-group to push them to be full
#   copt 4x distribution, but make all + the same
def bid_malicious__eager_to_same(B_old_malicious, SA_new, targets):
    B_new = copy_bids(len(targets), B_old_malicious, SA_new, strategy_map[1], targets)
    B_new[:, targets] = 1
    positive_bids = B_new[0, :] == 1 
    B_new[B_new == 1] = 0
    B_new[:, positive_bids] = 1
    return B_new

# indifferent to similar non-group
def bid_malicious__indifferent_to_similar(B_old_malicious, SA_new, targets):
    B_new = copy_bids(len(targets), B_old_malicious, SA_new, strategy_map[2], targets)
    B_new[:, targets] = 1
    return B_new

# notwilling to similar non-group
def bid_malicious__notwilling_to_similar(B_old_malicious, SA_new, targets):
    B_new = copy_bids(len(targets), B_old_malicious, SA_new, strategy_map[3], targets)
    B_new[:, targets] = 1
    return B_new

# + on 1 target, 0 on others
def bid_malicious__eager_cycle(B_old_malicious, SA_new, targets):
    assert len(targets) > 1
    B_new = copy_bids(len(targets), B_old_malicious, SA_new, strategy_map[4], targets)
    for rid, target in enumerate(targets):
        i = rid + 1 if rid < len(targets) else 0
        B_new[i, target] = 1
    return B_new

# (maybe) don't implement: uncommon, and can't recreate how they judge which papers are popular --> just be optimistic about it
# copy 4x dist, but +/- goes only on unpop
# popularity: rank by number of eager bids made by other reviewers
# implement later
def bid_malicious__indifferent_to_popular(B_old_malicious, SA_new, targets):
    pass

# 'main' function for synthetic part
def prepare_experiment(nrev1, npap1, strategy, group_size):
    # load data here
    data = np.load('../../parse/analysis/Biddings.npz')
    B_old_honest, B_old_malicious = data['HB'], data['MB']

    with open('../../parse/analysis/maps.pkl', 'rb') as f:
        data = pickle.load(f)
        reviewer_to_sas = data['reviewer_to_sas']
        paper_to_sa = data['paper_to_sa']
        group_map = data['group_map']
        authored_sa_map = data['authored_sa_map']

    # pass into other fns
    SA_new, M_new, malicious, targets = construct_SA_new(nrev1, npap1, group_size, reviewer_to_sas, paper_to_sa, group_map, authored_sa_map)
    B_new = construct_honest_bid_matrix(B_old_honest, SA_new)
    fns = [bid_malicious__basic, bid_malicious__eager_to_same, bid_malicious__indifferent_to_similar, bid_malicious__notwilling_to_similar, bid_malicious__eager_cycle]
    B_new_malicious = fns[strategy](B_old_malicious, S_new, targets)
    for i, rid in enumerate(malicious):
        B_new[rid, :] = B_new_malicious[i, rid]

