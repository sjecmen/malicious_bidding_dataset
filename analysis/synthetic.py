import numpy as np
from utils import make_SA_matrix, similarity
from analysis import simple_detect, cluster_detect, low_rank_detect
import pickle
from LP import match

rng = None

def copy_bid_distribution(B_row, SA_old_row, SA_row, targets=[]):
    def random_bid_subset(B, papers, npos, nneg):
        nneg = min(nneg, len(papers) - npos) # clip neg if too many
        sequence = rng.permutation(papers)
        B[sequence[:npos]] = 1
        B[sequence[npos:npos+nneg]] = -1

    npap0 = B_row.size
    npap1 = SA_row.size

    papers_sim = [x[0] for x in np.argwhere(SA_row > 0)]
    papers_non = [x[0] for x in np.argwhere(SA_row == 0)]
    papers_sim_old = [x[0] for x in np.argwhere(SA_old_row > 0)]
    papers_non_old = [x[0] for x in np.argwhere(SA_old_row == 0)]

    # npos0/nneg0 does not count any bids on targets because B_row has zerod them
    pos = B_row == 1
    neg = B_row == -1
    npos0 = [np.sum(pos[papers_sim_old]), np.sum(pos[papers_non_old])]
    nneg0 = [np.sum(neg[papers_sim_old]), np.sum(neg[papers_non_old])]
    npos1 = npos0 # array len 2
    nneg1 = [int(np.round(x * (npap1 / npap0))) for x in nneg0] # array len2

    B_new = np.zeros(npap1)
    random_bid_subset(B_new, papers_sim, npos1[0], nneg1[0])
    random_bid_subset(B_new, papers_non, npos1[1], nneg1[1])
    return B_new

def copy_bids(nrev, B_old, SA_old, SA_new, models, targets=[]):
    assert len(targets) == 0 or len(targets) == nrev
    B_new = np.zeros((nrev, SA_new.shape[1]))
    for rid in range(nrev):
        model_rev = rng.choice(models)
        B_row = B_old[model_rev, :]
        SA_old_row = SA_old[model_rev, :]
        SA_row = SA_new[rid, :]
        B_new[rid, :] = copy_bid_distribution(B_row, SA_old_row, SA_row, targets)
    return B_new

def construct_SA_new(nrev1, npap1, group_size, reviewer_to_sas, reviewer_to_authored_sa, group_map):
    assert nrev1 == npap1 # for now
    groups_of_size = [x for _, x in group_map.items() if len(x) == group_size]
    model_group = rng.choice(groups_of_size)
    models = rng.choice(list(reviewer_to_sas.keys()), nrev1 - group_size)

    new_sa_to_papers = {}
    new_reviewer_to_sas = {}
    for rid, model in enumerate(list(model_group) + list(models)):
        new_reviewer_to_sas[rid] = reviewer_to_sas[model]
        paper_sa = reviewer_to_authored_sa[model]
        if paper_sa not in new_sa_to_papers:
            new_sa_to_papers[paper_sa] = {rid}
        else:
            new_sa_to_papers[paper_sa].add(rid)
    M = np.eye(nrev1)
   
    SA = make_SA_matrix(nrev1, npap1, new_reviewer_to_sas, new_sa_to_papers)
    return SA, M, list(range(group_size)), list(range(group_size))

def construct_honest_bid_matrix(B_old_honest, SA_old, SA_new):
    nrev0, npap0 = B_old_honest.shape
    nrev1, npap1 = SA_new.shape
    B_new = copy_bids(nrev1, B_old_honest, SA_old, SA_new, list(range(nrev0)))
    return B_new

def bid_malicious__basic(B_old_malicious, SA_old, SA_new, strategy_map, targets):
    assert len(targets) == SA_new.shape[0]
    B_new = copy_bids(len(targets), B_old_malicious, SA_old, SA_new, strategy_map[0], targets)
    B_new[:, targets] = 1
    return B_new

def bid_malicious__notwilling_to_similar(B_old_malicious, SA_old, SA_new, strategy_map, targets):
    B_new = copy_bids(len(targets), B_old_malicious, SA_old, SA_new, strategy_map[1], targets)
    B_new[:, targets] = 1
    return B_new

def bid_malicious__eager_to_same(B_old_malicious, SA_old, SA_new, strategy_map, targets):
    B_new = copy_bids(len(targets), B_old_malicious, SA_old, SA_new, strategy_map[2], targets)
    B_new[:, targets] = 1
    positive_bids = B_new[0, :] == 1 
    B_new[B_new == 1] = 0
    B_new[:, positive_bids] = 1
    return B_new

def bid_malicious__eager_cycle(B_old_malicious, SA_old, SA_new, strategy_map, targets):
    assert len(targets) > 1
    B_new = copy_bids(len(targets), B_old_malicious, SA_old, SA_new, strategy_map[3], targets)
    for rid, target in enumerate(targets):
        i = rid + 1 if rid + 1 < len(targets) else 0
        B_new[i, target] = 1
    return B_new


# TODO refactor to pass around data map instead
def prepare_experiment(nrev1, npap1, strategy, group_size, data):
    # load data here
    B_old_honest, B_old_malicious, SA_old = data['HB'], data['MB'], data['SA']
    reviewer_to_sas = data['reviewer_to_sas']
    group_map = data['group_map']
    authored_sa_map = data['authored_sa_map']
    strategy_map = data['strategy_to_reviewers']

    # 0 honest bids on own paper so it is not counted in copy
    author_map = data['author_map']
    for rid, pid in author_map.items():
        B_old_honest[rid, pid] = 0
    # 0 malicious bids on target papers so they are not counted in copy
    author_map_group = data['author_map_group']
    for rid, pids in author_map_group.items():
        for pid in pids:
            B_old_malicious[rid, pid] = 0
    target_map = data['target_map'] # lone reviewer targets
    for rid, pid in target_map.items():
        B_old_malicious[rid, pid] = 0

    # pass into other fns
    SA_new, M_new, malicious, targets = construct_SA_new(nrev1, npap1, group_size, reviewer_to_sas, authored_sa_map, group_map)
    B_new = construct_honest_bid_matrix(B_old_honest, SA_old, SA_new)
    fns = [bid_malicious__basic, bid_malicious__notwilling_to_similar, bid_malicious__eager_to_same, bid_malicious__eager_cycle]
    SA_new_malicious = SA_new[malicious, :]
    B_new_malicious = fns[strategy](B_old_malicious, SA_old, SA_new_malicious, strategy_map, targets)
    for i, rid in enumerate(malicious):
        B_new[rid, :] = B_new_malicious[i, :]
    S = similarity(SA_new, B_new)
    return S, B_new, M_new, malicious, targets


def synth_bid_success(nrev, npap, strategy, group_size, data, num_trials):
    successes = []
    for t in range(num_trials):
        print('success trial', t)
        S, B, M, malicious, targets = prepare_experiment(nrev, npap, strategy, group_size, data)
        A = match(S, M)
        success_rate = 0
        for i in malicious:
            v = np.sum(A[i, targets])
            if v >= 1:
                success_rate += 1
            else:
                assert v == 0
        successes.append(success_rate)
    return successes 

def synth_bid_detect(nrev, npap, strategy, group_size, data, num_trials, detection_type, rank=None):
    reviewer_ranks = []
    for t in range(num_trials):
        print('detect trial', t)
        S, B, M, malicious, targets = prepare_experiment(nrev, npap, strategy, group_size, data)
        if detection_type == 'simple':
            detection_ranks = simple_detect(B, M)
        elif detection_type == 'low_rank':
            detection_ranks = low_rank_detect(B, M, rank)
        elif detection_type == 'cluster':
            author_map = {}
            for rid, pid in np.argwhere(M == 1):
                author_map[rid] = pid
            detection_ranks = cluster_detect(B, M, author_map)
        else:
            assert(False)

        these_ranks = [detection_ranks[i] for i in malicious]
        reviewer_ranks.append(these_ranks)
    return reviewer_ranks

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    with open('data/maps.pkl', 'rb') as f:
        data = pickle.load(f)
    data_load = np.load('data/Biddings.npz')
    for key, value in data_load.items():
        data[key] = value

    n = 5000 # number of reviewers and papers
    group_size = 4
    num_trials = 10
    success_by_strategy = [None] * 4
    rank_by_strategy_simple = [None] * 4
    rank_by_strategy_low_rank = [None] * 4
    rank_by_strategy_cluster = [None] * 4
    for strategy in range(4): # s4 not implemented
        print('strategy', strategy)
        success_by_strategy[strategy] = synth_bid_success(n, n, strategy, group_size, data, num_trials)
        rank_by_strategy_simple[strategy] = synth_bid_detect(n, n, strategy, group_size, data, num_trials, 'simple', rank=None)
        rank_by_strategy_cluster[strategy] = synth_bid_detect(n, n, strategy, group_size, data, num_trials, 'cluster', rank=None)
        rank_by_strategy_low_rank[strategy] = synth_bid_detect(n, n, strategy, group_size, data, num_trials, 'low_rank', rank=3)

    np.savez('data/Result_synth.npz', success_by_strategy=success_by_strategy, rank_by_strategy_simple=rank_by_strategy_simple,
            rank_by_strategy_cluster=rank_by_strategy_cluster, rank_by_strategy_low_rank=rank_by_strategy_low_rank, 
            n=n, group_size=group_size, num_trials=num_trials)
