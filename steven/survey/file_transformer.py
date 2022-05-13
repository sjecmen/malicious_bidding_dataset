import csv

# converts the group csv to the file to upload to qualitrics

# initial sheet with group and sa assignments, with emails in name col
group_sheet_name = 'new_groups.csv'
outfile_name = 'qualitrics_' + group_sheet_name

subject_area_names = ["Humans and AI: Fairness",
"Humans and AI: Interpretability, accountability, and transparency in AI",
"Social choice theory: Voting rules",
"Social choice theory: Ranking models",
"Social choice theory: Mechanism design",
"Game theory: Cooperative games",
"Game theory: Non-cooperative games",
"Game theory: Stackelberg security games",
"Game theory: No-regret learning",
"Probabilistic modeling: Bayesian theory",
"Probabilistic modeling: Graphical models",
"Probabilistic modeling: Gaussian processes",
"Probabilistic modeling: Topic models",
"Search: A* algorithms and variants",
"Search: Robotics applications of search",
"Optimization: Integer programming",
"Optimization: Linear programming",
"Optimization: Convex optimization",
"Optimization: Non-convex optimization",
"Machine learning: Classical machine learning (SVM, kernel methods)",
"Machine learning: Deep learning",
"Machine learning: Computer vision",
"Machine learning: Natural language processing",
"Machine learning: Bias-complexity tradeoffs and double descent curves",
"Machine learning: Learning theory"]

paper_names = ["Making Machine Learning Fair",
"Interpreting AI Decision-Making",
"Transparent Machine-Learning Models",
"Voting for Participatory Budgeting",
"Voting with Delegation",
"Improved Learning from Rankings",
"Equilibrium Selection in Cooperative Games",
"Communication for Teamwork in Games",
"Multi-Agent Cooperative Board Games",
"An Overview of Zero-Sum Games",
"Solution Concepts in Many-Player Games",
"Stackelberg Security Games for Coastal Defense",
"A No-Regret Algorithm for Efficient Equilibrium Computation",
"Consistency of Bayesian Inference",
"Fitting Graphical Models",
"A Novel Gaussian Process Approximation",
"A*+DFS: A Hybrid Search Algorithm",
"A* Search Under Uncertainty",
"Search-Based Planning of Robot Trajectories",
"Online Convex Optimization with Regularization",
"Private Stochastic Convex Optimization",
"A Hybrid Method for Non-Convex Optimization",
"Memory and Computation-Efficient Kernel SVMs",
"Forecasting Stock Prices with Deep Learning",
"A Deep Learning Approach for Anomaly Detection",
"New Algorithms for 3D Computer Vision",
"Towards More Accurate NLP Models",
"Optimal Error Bounds in Statistical Learning Theory"]

with open(group_sheet_name, newline='') as infile, open(outfile_name, mode='w') as outfile:
    r = csv.reader(infile)
    old_header = next(r)
    assert(all([x == y for x, y in 
        zip(old_header, ['name', 'sas', 'authored_sa', 'authored_id', 'target_sa', 'target_id', 'group'])
        ]))
    new_header = ['email', 'subject_area1', 'subject_area2', 'subject_area3', 'authored_paper', 'target_paper', 'group_papers']
    counts = [0] * len(paper_names)
    group_to_authored = {}
    new_rows = []
    groups = []
    for row in r:
        name = row[0].strip() # actually the email
        sas = [int(x) for x in row[1].strip().split(' ')]
        authored_id = int(row[3])
        target_id = row[5]
        group = int(row[6])

        new_row = [name]

        assert(len(sas) == 3) 
        for sa_id in sas:
            sa_name = subject_area_names[sa_id]
            new_row.append(sa_name)
            
        authored_name = paper_names[authored_id]
        new_row.append(authored_name)
        counts[authored_id] += 1

        target_name = '' if len(target_id) == 0 else paper_names[int(target_id)]
        new_row.append(target_name)
        
        if group in group_to_authored:
            group_to_authored[group].append(authored_name)
        else:
            group_to_authored[group] = [authored_name]
  
        new_rows.append(new_row)
        groups.append(group)
    assert(all([i == 2 for i in counts]))

    for new_row, group in zip(new_rows, groups):
        group_papers = group_to_authored[group].copy()
        authored_paper = new_row[4]
        group_papers.remove(authored_paper)
        group_paper_string = '; '.join(group_papers)
        new_row.append(group_paper_string)

    w = csv.writer(outfile)
    w.writerow(new_header)
    w.writerows(new_rows)
