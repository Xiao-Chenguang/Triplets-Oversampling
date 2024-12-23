from copy import deepcopy

import torch


@torch.no_grad()
def compute_updat(current_model, previous_model):
    # complete the function to return the difference of twe model
    update = deepcopy(current_model)
    for u, c, p in zip(
        update.parameters(), current_model.parameters(), previous_model.parameters()
    ):
        u.mul_(0)
        u.add_(c - p)
    return update


def compute_updat_v0(current_model, previous_model):
    # complete the function to return the difference of twe model
    current_state = current_model.state_dict()
    previous_state = previous_model.state_dict()
    for k in current_state:
        current_state[k] -= previous_state[k]
    return current_model


@torch.no_grad()
def weight_sum(updates, weights):
    # complete the weighted sum function.
    grad = deepcopy(updates[0])
    for k, v in grad.named_parameters():
        v.mul_(weights[0])
        for update, weight in zip(updates[1:], weights[1:]):
            v.add_(update.state_dict()[k], alpha=weight)
    return grad


@torch.no_grad()
def add_state(state, updates, a, b):
    if state is None:
        return deepcopy(updates)
        # complete the function to add the updates to the state
    for x, y in zip(state.parameters(), updates.parameters()):
        x.mul_(a)
        x.add_(y.mul(b))
    return state
