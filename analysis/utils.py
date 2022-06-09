import numpy as np

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
l1_subject_area_names = ["Humans and AI", "Social choice theory", 
                         "Game theory", "Probabilistic modeling", 
                         "Search", "Optimization", "Machine learning"]

# index of subject area to the high-level topic index
def l2_to_l1(l2):
    return l1_subject_area_names.index(subject_area_names[l2].split(":")[0])

# index of high-level topic to set of subject area indices
def l1_to_l2(l1):
    return {i for i, name in enumerate(subject_area_names) 
            if name.split(":")[0] == l1_subject_area_names[l1]
           }

def make_SA_matrix(nrev, npap, reviewer_to_sas, sa_to_papers):
    SA = np.zeros((nrev, npap))
    for reviewer_id in range(len(reviewer_to_sas)):
        l2_sas = reviewer_to_sas[reviewer_id]
        l1_sas = {l2_to_l1(x) for x in l2_sas}
        # set matching high-level topic to 0.5
        for l1_sa in l1_sas:
            for sa_id in l1_to_l2(l1_sa): # some SAs don't have papers
                if sa_id not in sa_to_papers:
                    continue
                for paper_id in sa_to_papers[sa_id]:
                    SA[reviewer_id, paper_id] = 0.5
        # set matching subject areas to 1 (overwriting)
        for sa_id in l2_sas:
            if sa_id not in sa_to_papers:
                continue
            for paper_id in sa_to_papers[sa_id]:
                SA[reviewer_id, paper_id] = 1
    return SA

def similarity(SA, B):
    return (1+SA) * np.power(2, B)
