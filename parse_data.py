import csv
import random
import string
import time
import numpy as np
import pickle
from analysis.utils import *

# Parse dataset CSVs into format for analysis

nrev = 31
npap = 28

email_to_id = {}
paper_to_id = {}
strategy_descriptions = {}
HB = np.zeros((nrev, npap)) # honest bidding matrix with entries in {-1, 0, 1}
MB = np.zeros((nrev, npap)) # malicious bidding matrix with entries in {-1, 0, 1}

with open('dataset/malicious_bidding.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    bid_start_idx = 1
    assert(header[0] == 'Name' and header[bid_start_idx] == 'Q3_1' and
           header[bid_start_idx + npap - 1] == 'Q3_28')
    header = next(csv_reader)
    assert header[0] == 'Name'
    for paper_id, paper in enumerate(header[bid_start_idx:bid_start_idx+npap]):
        paper_name = paper[paper.find(': -') + 3:].strip()
        paper_to_id[paper_name] = paper_id
    reviewer_id = 0 # assign reviewer_ids only to the subset of malicious bids
    for row in csv_reader:
        email_to_id[row[0]] = reviewer_id
        strategy_descriptions[reviewer_id] = row[bid_start_idx + npap]
        for paper_id in range(npap):
            if row[bid_start_idx + paper_id] == 'Not willing to review':
                MB[reviewer_id, paper_id] = -1
            elif row[bid_start_idx + paper_id] == 'Indifferent':
                MB[reviewer_id, paper_id] = 0
            elif row[bid_start_idx + paper_id] == 'Eager to review':
                MB[reviewer_id, paper_id] = 1
        reviewer_id += 1
    assert reviewer_id == nrev, reviewer_id

with open('dataset/honest_bidding.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    assert(header[0] == 'Name' and header[bid_start_idx] == 'Q3_1' and
           header[bid_start_idx + npap - 1] == 'Q3_28')
    header = next(csv_reader)
    assert header[0] == 'Name'
    for row in csv_reader:
        if row[0] not in email_to_id: # did not submit malicious bids
            continue
        reviewer_id = email_to_id[row[0]]
        for paper_id in range(npap):
            if row[bid_start_idx + paper_id] == 'Not willing to review':
                HB[reviewer_id, paper_id] = -1
            elif row[bid_start_idx + paper_id] == 'Indifferent':
                HB[reviewer_id, paper_id] = 0
            elif row[bid_start_idx + paper_id] == 'Eager to review':
                HB[reviewer_id, paper_id] = 1

strategy_map = {}
strategy_to_reviewers = {}
with open('dataset/strategy_annotations.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        if row[0] not in email_to_id: # did not submit malicious bids
            continue
        reviewer_id = email_to_id[row[0]]
        strat = int(row[1])
        assert strat < 5
        strategy_map[reviewer_id] = strat
        if strat not in strategy_to_reviewers:
            strategy_to_reviewers[strat] = [reviewer_id]
        else:
            strategy_to_reviewers[strat].append(reviewer_id)


reviewer_to_sas = {} # dict of review -> list of subject areas
reviewer_to_group = {} # dict of reviewer -> group id
reviewer_to_sa = {} # dict of review -> subject area of authored paper
sa_to_paper_ids = {} # dict of subject area -> list of papers in the subject area
paper_to_sa = {} # dict of paper -> subject area
group_to_targets = {} # dict of group -> list of target papers

group_map = {} # dict of group id -> list of reviewer indices in that group
author_map = {} # dict of reviewer -> index of paper authored
target_map = {} # dict of reviewer -> index of target paper (for reviewers with group size 1)
author_map_group = {} # dict of reviewer -> indices of target papers (including other team members' authored papers)

with open('dataset/setup.csv') as csvfile:
    r = csv.reader(csvfile)
    header = next(r)
    assert(all([x == y for x, y in 
        zip(header, ['name', 'sas', 'authored_sa', 'authored_id', 'target_sa', 'target_id', 'group'])
        ]))
    for row in r:
        if row[0] in email_to_id:
            reviewer_id = email_to_id[row[0]]
        else:
            reviewer_id = -1
        
        # sas and sa_id are subject area indices
        sas = {int(x) for x in row[1].strip().split(' ')}
        sa_id = int(row[2]) 
        # authored_id is a paper index into the list, but assigned paper_id based on survey position
        paper_id = paper_to_id[paper_names[int(row[3])]]
        
        # store which papers are in which subject areas
        paper_to_sa[paper_id] = sa_id
        if sa_id in sa_to_paper_ids:
            sa_to_paper_ids[sa_id].add(paper_id)
        else:
            sa_to_paper_ids[sa_id] = {paper_id}
        
        if reviewer_id >= 0:
            reviewer_to_sas[reviewer_id] = sas
            reviewer_to_sa[reviewer_id] = sa_id
            author_map[reviewer_id] = paper_id 
            
            group = int(row[6])
            if group in group_map:
                group_map[group].append(reviewer_id)
            else:
                group_map[group] = [reviewer_id]
            reviewer_to_group[reviewer_id] = group
            # if group size 1, read target paper subject area and paper indices
            if len(row[5]) > 0: 
                target_id = paper_to_id[paper_names[int(row[5])]]
                target_map[reviewer_id] = target_id
        
        group = int(row[6])
        if group in group_to_targets:
            group_to_targets[group].append(paper_id)
        else:
            group_to_targets[group] = [paper_id]
            
# compute SA matrix
SA = make_SA_matrix(nrev, npap, reviewer_to_sas, sa_to_paper_ids)

for reviewer_id in range(len(reviewer_to_group)):
    author_map_group[reviewer_id] = group_to_targets[reviewer_to_group[reviewer_id]]


np.savez('analysis/data/Biddings.npz', HB=HB, MB=MB, SA=SA)

with open('analysis/data/maps.pkl', 'wb') as f:
    pickle.dump({
        'author_map' : author_map, 
        'group_map' : group_map, 
        'target_map' : target_map, 
        'author_map_group' : author_map_group, 
        'authored_sa_map' : reviewer_to_sa,
        'reviewer_to_sas' : reviewer_to_sas,
        'paper_to_sas' : paper_to_sa,
        'reviewer_to_strategy' : strategy_map,
        'strategy_to_reviewers' : strategy_to_reviewers,
        'strategy_descriptions' : strategy_descriptions
    }, f)
