import numpy as np
from LP import match
import random
import scipy.linalg
import pickle
from utils import similarity

'''
Arguments used for various functions (inputs from the survey):
    HB : np matrix of size (#reviewers, #papers) with entries in {-1, 0, 1}
        denoting honest bids
    MB : as above, but containing manipulating bids
    SA: np matrix of size (#reviewers, #papers) with entries in {0, 1}
        denoting if the subject area matches
    group_map : dict of group id (arbitrary) -> list of reviewer indices in that group
    author_map : dict of reviewer index -> index of paper authored
    target_map : dict of reviewer index -> index of target paper (for reviewers with group size 1)
'''

# MAIN FUNCTIONS

'''
Determine whether manipulation was successful.
Inputs:
    SA, HB, MB, authored_map, target_map, group_map
    num_trials : number of trials to run for each manipulating group (due to randomness
        in the subsampling of honest reviewers)
    authored_map_group: list of papers authored by group members
        including members who didn't participated the game
Outputs:
    successes_by_rev : np vector of size (#reviewers)
        denoting the number of target papers that each reviewer got assigned to
'''
def bid_success(SA, HB, MB, authored_map, authored_map_group, target_map, group_map, num_trials):
    full_M = construct_conflicts(authored_map, SA.shape)
    successes_by_rev = np.zeros(SA.shape[0])
    for group_id, group_revs in group_map.items():
        target_paps = authored_map_group[group_revs[0]] # groups target their papers
        if len(group_revs) == 1 and group_revs[0] in target_map:
            target_paps = [target_map[group_revs[0]]] # lone revs have a separate target
        for t in range(num_trials):
            S, B, M, idx_map = construct_similarity(SA, HB, MB, full_M, group_revs)
            A = match(S, M)
            for i in group_revs:
                new_i = idx_map[i]
                v = np.sum(A[new_i, target_paps])
                if v >= 1:
                    successes_by_rev[i] += 1
                else:
                    assert v == 0
    return successes_by_rev


'''
Determine whether reviewers were detected.
Inputs:
    SA, HB, MB, authored_map, group_map,
    num_trials : number of trials to run for each manipulating group (due to randomness
        in the subsampling of honest reviewers)
    detection_type : string in {"simple", "low_rank", "cluster"} denoting which
        detection algorithm to run
    rank : integer denoting what rank the bid matrix should be thresholded to, only
        if detection_type == "low_rank"
Outputs:
    rank_by_rev : list of size (#reviewers)
        denoting the list of ranks given to each reviewer by the detection algorithm
        (lower rank means more suspicious)
'''
def bid_detect(SA, HB, MB, authored_map, group_map, num_trials, detection_type, rank=None):
    full_M = construct_conflicts(authored_map, SA.shape)
    #rank_by_rev = np.zeros(SA.shape[0])
    rank_by_rev = [[] for _ in range(SA.shape[0])]
    for group_id, group_revs in group_map.items():
        for t in range(num_trials):
            _, B, M, idx_map = construct_similarity(SA, HB, MB, full_M, group_revs)

            if detection_type == 'simple':
                detection_ranks = simple_detect(B, M)
            elif detection_type == 'low_rank':
                detection_ranks = low_rank_detect(B, M, rank)
            elif detection_type == 'cluster':
                detection_ranks = cluster_detect(B, M, authored_map)
            else:
                assert(False)

            for i in group_revs:
                new_i = idx_map[i]
                position = detection_ranks[new_i]
                rank_by_rev[i].append(position)
    return rank_by_rev

'''
Calculates the difference between the honest and manipulated bids for each reviewer
Inputs:
    HB, MB
    M : conflict matrix from construct_conflicts
    diff_fn : string in {'hamming', 'L1'} denoting which function to
        calculate differences with
Outputs:
    np vector of size (#reviewers) denoting the difference between honest and manipulated
        bids for each reviewer
'''
def bid_difference(HB, MB, author_map, diff_fn):
    M = construct_conflicts(author_map, HB.shape)
    HB[M == 1] = 0
    MB[M == 1] = 0
    if diff_fn == 'hamming':
        return np.sum(HB != MB, axis=1)
    elif diff_fn == 'L1':
        return np.sum(np.abs(HB - MB), axis=1)
    else:
        assert(False)


# SUBROUTINES

'''
Construct conflict matrix according to authorship
Inputs:
    author_map
    shape : tuple of (#reviewers, #papers)
Outputs:
    M : np matrix of size (#reviewers, #papers) in {0, 1}
        denoting if the reviewer authored the paper
'''
def construct_conflicts(author_map, shape):
    M = np.zeros(shape)
    for rev, authored_pap in author_map.items():
        M[rev, authored_pap] = 1
    return M

'''
Subsample honest reviewers and apply similarity function to construct
a sample square similarity matrix for use in assignment or detection. The
resulting matrix contains all manipulating reviewers and a random choice
of honest reviewers so that the number of reviewers equals the number of papers.
Inputs:
    SA, HB, MB, M
    manipulators : list of reviewer indices of manipulating reviewers
Outputs:
    S : np matrix of size (#papers, #papers) where entry (r, p) denotes
        the similarity between reviewer r (using a new index) and paper p
    B : np matrix of size (#papers, #papers) where entry (r, p) denotes
        the bid of reviewer r (using a new index) on paper p
    newM : np matrix of size (#papers, #papers) that contains the conflicts
        from M, using the new reviewer indices
    idx_map : dict mapping the old reviewer index (used for SA, HB, MB, M)
        to the new reviewer index (used for S, B, newM)
'''
def construct_similarity(SA, HB, MB, M, manipulators):
    nrev, npap = HB.shape
    assert(nrev >= npap) # need at least enough reviewers to fill matrix
    revs = [r for r in range(nrev) if r not in manipulators]
    nhonest = npap - len(manipulators)
    honest_revs = random.sample(revs, nhonest)
    new_idxs = manipulators + honest_revs # old idx in its new position
    random.shuffle(new_idxs) # so that later tiebreaking is random
    idx_map = {old_i : new_i for new_i, old_i in enumerate(new_idxs)} # old idx to new

    B = np.zeros((npap, npap))
    for new_i, old_i in enumerate(new_idxs):
        if old_i in manipulators:
            B[new_i, :] = MB[old_i, :]
        else:
            B[new_i, :] = HB[old_i, :]
    newSA = SA[new_idxs, :]
    newM = M[new_idxs, :]
    S = similarity(newSA, B)
    return S, B, newM, idx_map



# DETECTION ALGORITHMS
# These all return maps from the reviewer index returned by construct_similarity to their
# rank in order of suspiciousness. I.e., a reviewer ranked k means the algo thinks there
# are k more suspicious reviewers than them.

'''
Detect bids by the difference in the number of positive and negative bids.
Inputs:
    B, M : subsampled bid and conflict matrices from construct_similarity
Outputs:
    sorted_revs : map of (new) reviewer index to their rank in order of suspiciousness
'''
def simple_detect(B, M):
    B[M == 1] = 0
    revs = list(range(B.shape[0]))
    diffs = []
    for r in revs:
        bids = B[r, :]
        num_pos = np.sum(bids == 1)
        num_neg = np.sum(bids == -1)
        diffs.append(num_neg - num_pos) # larger diff is more sus
    sorted_revs = {x : i for i, (_, x) in enumerate(sorted(zip(diffs, revs), reverse=True))} # map rev to position
    return sorted_revs

'''
Detect pairs where both reviewers bid on each other's paper, then pairs where one reviewer bids on the other's.
Ties broken by how these reviewers bid on other papers (where a higher total bid value on outside papers means they are less suspicious).
Inputs:
    B, M : subsampled bid and conflict matrices from construct_similarity
    authored_map
Outputs:
    sorted_revs : map of (new) reviewer index to their rank in order of suspiciousness
'''
def cluster_detect(B, M, authored_map):
    B[M == 1] = 0
    nrev, npap = B.shape
    scores = np.zeros(B.shape)
    bidsums = np.sum(B, axis=1)
    for i in range(nrev):
        for j in range(i):
            pi = authored_map[i]
            pj = authored_map[j]
            i_out = bidsums[i] - B[i, pj] # all bids other than pj
            j_out = bidsums[j] - B[j, pi]
            # sort revs first by in-group bids, then by out-group
            score = max(B[i, pj], 0) + max(B[j, pi], 0) - ((i_out + j_out) / (2 * npap))
            scores[i, j] = score
            scores[j, i] = score
    rev_scores = list(np.amax(scores, axis=1))
    sorted_revs = {x : i for i, (_, x) in enumerate(sorted(zip(rev_scores, list(range(B.shape[0]))), reverse=True))} # map rev to position
    return sorted_revs

'''
Detect bids by the difference from the low rank approximation to the bidding matrix.
Inputs:
    B, M : subsampled bid and conflict matrices from construct_similarity
    rank : rank to threshold the bidding matrix to
    method : string in {'max', 'sum'} denoting how to determine the overall
        suspiciousness of a reviewer from the suspiciousness of their individual bids on each paper
        - method='max': Rank reviewers by their most different bid
        - method='sum': Rank reviewers by the sum of bid differences
Outputs:
    sorted_revs : map of (new) reviewer index to their rank in order of suspiciousness
'''
def low_rank_detect(B, M, rank, method='sum'):
    assert(rank != None)
    B[M == 1] = 0
    L = threshold(B, rank)
    delta = np.abs(L - B)
    delta[M == 1] = 0 # don't factor in predicted bids on own papers
    if method == 'sum':
        diffs = list(np.sum(delta, axis=1))
    elif method == 'max':
        diffs = list(np.amax(delta, axis=1))
    else:
        assert(False)
    sorted_revs = {x : i for i, (_, x) in enumerate(sorted(zip(diffs, list(range(B.shape[0]))), reverse=True))} # map rev to position
    return sorted_revs

'''
Threshold low singular values of S so it has given rank.
Inputs:
    S : np matrix
    rank : desired rank
Outputs:
    np matrix thresholded to rank
'''
def threshold(S, rank):
    U, s, Vh = scipy.linalg.svd(S, full_matrices=False)
    #print('singular values:', s[:10])
    k = s.size
    if k < rank:
        return S
    for i in range(rank, k):
        s[i] = 0
    return U @ np.diag(s) @ Vh


def main():
    random.seed(0)
    data = np.load('data/Biddings.npz')
    HB, MB, SA = data['HB'], data['MB'], data['SA']
    
    with open('data/maps.pkl', 'rb') as f:
        data = pickle.load(f)
        author_map, group_map, target_map, author_map_group = (
            data['author_map'], data['group_map'], data['target_map'], data['author_map_group'])

    num_trials = 10
    success_by_reviewer = bid_success(SA, HB, MB, author_map, author_map_group, target_map, group_map, num_trials=num_trials)
    rank_by_reviewer_simple = bid_detect(SA, HB, MB, author_map, group_map, num_trials=num_trials, detection_type='simple', rank=None)
    rank_by_reviewer_cluster = bid_detect(SA, HB, MB, author_map, group_map, num_trials=num_trials, detection_type='cluster', rank=None)
    rank_by_reviewer_low_rank = bid_detect(SA, HB, MB, author_map, group_map, num_trials=num_trials, detection_type='low_rank', rank=3)

    np.savez('data/Result.npz', success_by_reviewer=success_by_reviewer, rank_by_reviewer_simple=rank_by_reviewer_simple,
            rank_by_reviewer_cluster=rank_by_reviewer_cluster, rank_by_reviewer_low_rank=rank_by_reviewer_low_rank, 
            num_trials=num_trials)

if __name__ == "__main__":
    main()
