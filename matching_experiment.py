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
        for j in range(arrival_counts[i].long().item()):
            results.append(i)
    return torch.LongTensor(results)

def step_simulation(current_elems, match_edges, type_arrival_rates, type_departure_probs, match_thresh=0.8):
    # first match elements
    pool_after_match = current_elems[torch.max(match_edges, 0).values <= match_thresh]
    
    # now handle departures
    if pool_after_match.shape[0] > 0:
        remaining_elements_depart_prob = type_departure_probs[pool_after_match]
        remain = torch.bernoulli(1 - remaining_elements_depart_prob).nonzero().view(-1)
        remaining_elements = pool_after_match[remain]
    else:
        remaining_elements = pool_after_match
    
    # now get new elements (poisson?)
    after_arrivals = torch.cat((remaining_elements, ind_counts_to_longs(torch.poisson(type_arrival_rates))))
    
    return after_arrivals

def edge_matrix(current_elems, e_weights_by_type):
    lhs_matrix = current_elems.repeat(current_elems.shape[0],1)
    rhs_matrix = lhs_matrix.t()
    return e_weights_by_type[lhs_matrix, rhs_matrix]

def compute_matching(current_elems, curr_type_weights, e_weights_by_type, gamma=0.000001):
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
    mat[0,1] = 10.0
    mat[1,0] = 10.0
    mat[0,0] = 0.0
    mat[0,2:5] = 0.0
    mat[2:5,0] = 0.0
    return mat

toy_arrival_rates = torch.Tensor([0.2,1.0,1.0,1.0,1.0])
toy_departure_probs = torch.Tensor([0.9,0.1,0.1,0.1,0.1])

def train_func(n_rounds=20, n_epochs=20):
    e_weights_type = toy_e_weights_type()
    init_pool = torch.LongTensor([1,1,2,2])
    type_weights = torch.full((5,), 0.05, requires_grad=True)
    optimizer = torch.optim.Adam([type_weights], lr=5e-2, weight_decay=1e-2)
    total_losses = []
    for e in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        losses = []
        curr_pool = init_pool.clone()
        for r in range(n_rounds):
            resulting_match, e_weights = compute_matching(curr_pool, type_weights, e_weights_type)
            print(resulting_match)
            losses.append(-1.0*torch.sum(resulting_match * e_weights))
            curr_pool = step_simulation(curr_pool, resulting_match, toy_arrival_rates, toy_departure_probs)
        total_loss = torch.sum(torch.stack(losses))
        total_losses.append(total_loss.item())
        total_loss.backward()
        optimizer.step()
    return type_weights, total_losses

def eval_func(trained_weights, n_rounds = 30, n_epochs=10):
    e_weights_type = toy_e_weights_type()
    type_weights = trained_weights.detach()
    init_pool = torch.LongTensor([1,1,2,2])
    all_losses = []
    for e in tqdm(range(n_epochs)):
        losses = []
        curr_pool = init_pool.clone()
        for r in range(n_rounds):
            resulting_match, e_weights = compute_matching(curr_pool, type_weights, e_weights_type)
            losses.append(-1.0*torch.sum(resulting_match * e_weights).item())
            curr_pool = step_simulation(curr_pool, resulting_match, toy_arrival_rates, toy_departure_probs)
        all_losses.append(losses)
    return all_losses

if __name__ == '__main__':
    result_weights, learning_loss = train_func()
    print(learning_loss)
    print(result_weights)
    loss_list = eval_func(result_weights, n_epochs=30)
    print('loss of learned weights:', np.mean(np.sum(loss_list, axis=1)))
    
    
    ones_loss_list = eval_func(torch.ones(5), n_epochs=30)
    print('loss of constant weights:', np.mean(np.sum(ones_loss_list,axis=1)))
