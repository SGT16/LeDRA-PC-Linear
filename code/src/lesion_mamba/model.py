import torch

def optimizer_for(model):
    regular, no_decay = ([], [])
    for parameter in model.parameters():
        if parameter.requires_grad:
            (no_decay if getattr(parameter, '_no_weight_decay', False) else regular).append(parameter)
    groups = [dict(params=regular, lr=0.001, weight_decay=0.0001, group_name='ordinary')]
    if no_decay:
        groups.append(dict(params=no_decay, lr=0.001, weight_decay=0.0, group_name='mamba_A_log_D_no_decay'))
    return torch.optim.AdamW(groups, lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
