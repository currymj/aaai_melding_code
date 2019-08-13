import torch
import torch.nn as nn
import random
import numpy as np
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from collections import defaultdict

# matching matrix here corresponds to constraints only.
# so for kidneys we need to make the "each node in <= 1 cycle" constraint
# also we want to force cycle variables to be positive, as in edges below.


def make_matching_matrix(n):
    
    # n is num elements?
    lhs = list(range(n))
    rhs = list(range(n, 2*n))
    
    # n_vars is 1 per possible edge?
    n_vars = len(lhs)*len(rhs)
    # n_constraints is 1 for each lhs, 1 for each rhs, 1 per edge?
    n_constraints = len(lhs) + len(rhs) + n_vars
    A = np.zeros((n_constraints, n_vars))
    b = np.zeros((n_constraints))
    curr_idx = 0
    edge_idx = {}
    # get an index per edge
    for u in lhs:
        for v in rhs:
            edge_idx[(u,v)] = curr_idx
            curr_idx += 1
    # A has rows of 2n elements, followed by n^2 edges
    # A has cols of n^2 edges (so A @ x where x is edges)
    for u in lhs:
        for v in rhs:
            # for u, flip on coefficient for only its outgoing edges
            A[u, edge_idx[(u,v)]] = 1
            # for v, flip on coefficient for only its incoming edges
            A[v, edge_idx[(u,v)]] = 1
            # for the edge itself, flip on a single -1 at its point only (- point must be <= 0 i.e. point must be positive)
            A[len(lhs)+len(rhs)+edge_idx[(u,v)], edge_idx[(u,v)]] = -1
    
    # each element can have only 1 edge turned on in x
    for u in lhs:
        b[u] = 1
    for u in rhs:
        b[u] = 1
    
    
    return A, b

def ind_counts_to_longs(arrival_counts):
    # optimize later
    results = []
    for i in range(arrival_counts.shape[0]):
        for j in range(arrival_counts[i]):
            results.append(i)
    return torch.LongTensor(results)


def generate_full_history(type_arrival_rates, type_departure_probs, max_t):
    # an element is a list of (type, start_time, end_time)
    # too bad we don't have mutable namedtuples here, and it's probably not
    # worth creating a tiny class
    all_elems = []
    curr_elems = []
    for t in range(max_t):
        # departures
        next_currelems = []
        for i in range(len(curr_elems)):
            v = curr_elems[i]
            departing = np.random.rand() <= type_departure_probs[v[0]]
            if departing:
                v[2] = t
            else:
                next_currelems.append(v)
        curr_elems = next_currelems

        arrival_types = ind_counts_to_longs(np.random.poisson(lam=type_arrival_rates))
        arrivals = [[x, t, -1] for x in arrival_types]
        all_elems.extend(arrivals)
        curr_elems.extend(arrivals)

    for v in curr_elems:
        v[2] = max_t
    for v in all_elems:
        assert(v[1] >= 0)
        assert(v[2] >= 0)

    return all_elems

def history_to_arrival_dict(full_history):
    result = defaultdict(list)
    for v in full_history:
        result[v[1]].append(v)
    return result


def arrivals_only(current_elems, t_to_arrivals, curr_t):
    return current_elems + t_to_arrivals[curr_t]


def step_simulation(current_elems, match_edges, t_to_arrivals, curr_t, match_thresh=0.8):
    unmatched_indices = (torch.max(match_edges, 0).values <= match_thresh).nonzero().flatten().numpy()

    # get locations of maxima
    # remove from current_elems if the maxima are <= match_threshold.

    pool_after_match = []
    for i in range(len(current_elems)):
        if i in unmatched_indices:
            pool_after_match.append(current_elems[i])
    
    remaining_elements = [] 
    for v in pool_after_match:
        if v[2] > curr_t:
            remaining_elements.append(v)

    # now get new elements (poisson?)
    after_arrivals = remaining_elements + t_to_arrivals[curr_t]
    
    return after_arrivals

def edge_matrix(current_elems, e_weights_by_type):
    lhs_matrix = current_elems.repeat(current_elems.shape[0],1)
    rhs_matrix = lhs_matrix.t()
    return e_weights_by_type[lhs_matrix, rhs_matrix]

def compute_matching(current_pool_list, curr_type_weights, e_weights_by_type, gamma=0.000001):
    # convert current_elems to LongTensor of correct form here
    current_elems = torch.tensor([x[0] for x in current_pool_list])
    n = current_elems.shape[0]
    A, b = make_matching_matrix(n)
    A = torch.from_numpy(A).float()
    b = torch.from_numpy(b).float()
    # for some reason we need this randomness to end up with an actual matching
    e_weights = edge_matrix(current_elems, e_weights_by_type)
    jitter_e_weights = e_weights + 1e-4*torch.rand(n,n)
    #e_weights = torch.rand(n,n)
    model_params_quad = make_gurobi_model(A.detach().numpy(), b.detach().numpy(), None, None, gamma*np.eye(A.shape[1]))
    func = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params_quad)
    
    Q_mat = gamma*torch.eye(A.shape[1])
    
    curr_elem_weights = curr_type_weights[current_elems]
    modified_edge_weights = jitter_e_weights - 0.5*(torch.unsqueeze(curr_elem_weights,0) + torch.unsqueeze(curr_elem_weights,1))
    # may need some negative signs
    resulting_match = func(Q_mat, -modified_edge_weights.view(-1), A, b, torch.Tensor(), torch.Tensor()).view(n,n)
    return resulting_match, e_weights


## start of toy problem

def toy_e_weights_type():
    mat = 0.1*torch.ones(5,5)
    mat[0,1] = 3.0
    mat[1,0] = 3.0
    mat[0,0] = -100.0
    mat[0,2:5] = -100.0
    mat[2:5,0] = -100.0
    return mat

toy_arrival_rates = torch.Tensor([0.2,1.0,1.0,1.0,1.0])
toy_departure_probs = torch.Tensor([0.9,0.05,0.1,0.1,0.1])


def compute_discounted_returns(losses, gamma=1.0):
    # inspired originally by facebook's reinforce example
    returns = []
    R = 0.0
    for r in losses[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    return returns

        
def train_func(n_rounds=50, n_epochs=20):
    e_weights_type = toy_e_weights_type()
    type_weights = torch.full((5,), 0.0, requires_grad=True)
    optimizer = torch.optim.Adam([type_weights], lr=1e-1, weight_decay=1e-1)
    total_losses = []
    for e in tqdm(range(n_epochs)):
        full_history = generate_full_history(toy_arrival_rates, toy_departure_probs, n_rounds)
        t_to_arrivals = history_to_arrival_dict(full_history)
        optimizer.zero_grad()
        losses = []
        curr_pool = []
        for r in range(n_rounds):
            if len(curr_pool) <= 1:
                curr_pool = arrivals_only(curr_pool, t_to_arrivals, r)
                continue
            resulting_match, e_weights = compute_matching(curr_pool, type_weights, e_weights_type)
            losses.append(1.0*torch.sum(e_weights * resulting_match))
            curr_pool = step_simulation(curr_pool, resulting_match, t_to_arrivals, r)
        total_loss = torch.sum(torch.stack(compute_discounted_returns(losses)))
        total_losses.append(total_loss.item())
        total_loss.backward()
        optimizer.step()
    return type_weights, total_losses

def eval_func(trained_weights, n_rounds = 50, n_epochs=100):
    e_weights_type = toy_e_weights_type()
    type_weights = trained_weights.detach()
    all_losses = []
    for e in tqdm(range(n_epochs)):
        full_history = generate_full_history(toy_arrival_rates, toy_departure_probs, n_rounds)
        t_to_arrivals = history_to_arrival_dict(full_history)
        losses = []
        curr_pool = []
        for r in range(n_rounds):
            if len(curr_pool) <= 1:
                curr_pool = arrivals_only(curr_pool, t_to_arrivals, r)
                continue
            resulting_match, e_weights = compute_matching(curr_pool, type_weights, e_weights_type)
            losses.append(1.0*torch.sum(resulting_match * e_weights).item())
            curr_pool = step_simulation(curr_pool, resulting_match, t_to_arrivals, r)
        if len(losses) == 0:
            losses.append(0.0)
        all_losses.append(losses)
    return all_losses


if __name__ == '__main__':
    results_list = []
    train_epochs = 30
    test_epochs = 50
    n_experiments = 5
    for i in range(n_experiments):
        print(i)
        result_weights, learning_loss = train_func(n_epochs=train_epochs)
        print(learning_loss)
        print(result_weights)
        loss_list = eval_func(result_weights, n_epochs=test_epochs)
        learned_loss = np.mean([np.sum(l) for l in loss_list])
        learned_std = np.std([np.sum(l) for l in loss_list])
        print('loss of learned weights:', learned_loss)
        print('std of learned weights:', learned_std)
        
        
        const_loss_list = eval_func(torch.full((5,), 0.0, requires_grad=False), n_epochs=test_epochs)
        const_loss = np.mean([np.sum(l) for l in const_loss_list])
        const_std = np.std([np.sum(l) for l in const_loss_list])
        print('loss of initial constant weights:', const_loss)
        print('std of initial constant weights:', const_std)
        results_list.append( (learned_loss, learned_std, const_loss, const_std) )

    for i in range(n_experiments):
        print('experiment', i)
        losses = results_list[i]
        learned_ci = 1.96 * losses[1] / np.sqrt(test_epochs)
        const_ci =  1.96 * losses[3] / np.sqrt(test_epochs)

        print(f"learned weights mean: {losses[0]} +/- {learned_ci}")
        print(f"constant weights mean: {losses[2]} +/- {const_ci}")
