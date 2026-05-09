import torch
import numpy as np

# Use the existing device logic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_reward(seq, actions, ignore_far_sim=False, temp_dist_thre=20):
    """
    Standard Reward for SAC: Dense shaping without num_picks dilution.
    """
    _seq = seq.detach().squeeze()
    _actions = actions.detach().squeeze()
    n = _seq.size(0)
    step_reward = torch.zeros(n, device=device)

    pick_idxs = torch.nonzero(_actions, as_tuple=False).squeeze()
    
    if pick_idxs.ndimension() == 0 and pick_idxs.numel() > 0:
        pick_idxs = pick_idxs.unsqueeze(0)
    num_picks = len(pick_idxs) if pick_idxs.numel() > 0 else 0

    if num_picks == 0:
        return step_reward 

    # 1. Global Diversity
    if num_picks == 1:
        reward_sim = torch.tensor(0., device=device)
    else:
        normed_seq = _seq / (_seq.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sim_mat = torch.matmul(normed_seq, normed_seq.t())
        sim_submat = sim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.view(-1, 1).expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            sim_submat[temp_dist_mat > temp_dist_thre] = 1.
        
        reward_sim = sim_submat.sum() / (num_picks * (num_picks - 1.))

    # 2. Global Representativeness
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat = torch.addmm(input=dist_mat, beta=1, mat1=_seq, mat2=_seq.t(), alpha=-2)
    dist_mat = dist_mat[:, pick_idxs]
    
    if len(dist_mat.shape) == 1:
        dist_mat = dist_mat.unsqueeze(dim=1)
    
    dist_mat = torch.min(dist_mat, 1, keepdim=True)[0]
    reward_rep = torch.exp(-dist_mat.mean())

    # 3. SAC Dense Shaping: No division by num_picks
    total_global_reward = (reward_sim + reward_rep) * 0.5
    step_reward[pick_idxs] = total_global_reward 
    
    return step_reward

def compute_reward_coff(seq, actions, ignore_far_sim=True, temp_dist_thre=20, 
                        div_coff=0.5, rep_coff=0.5):
    """
    Coefficient-based version optimized for SAC.
    """
    _seq = seq.detach().squeeze()
    _actions = actions.detach().squeeze()
    n = _seq.size(0)
    step_reward = torch.zeros(n, device=device)

    pick_idxs = _actions.nonzero().squeeze()
    if pick_idxs.ndimension() == 0 and pick_idxs.numel() > 0:
        pick_idxs = pick_idxs.unsqueeze(0)
    num_picks = len(pick_idxs) if pick_idxs.numel() > 0 else 0

    if num_picks == 0:
        return step_reward

    # Diversity
    if num_picks == 1:
        reward_div = torch.tensor(0., device=device)
    else:
        normed_seq = _seq / (_seq.norm(p=2, dim=1, keepdim=True) + 1e-8)
        dissim_mat = 1 - torch.matmul(normed_seq, normed_seq.t())
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.view(-1, 1).expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

    # Representativeness
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat = torch.addmm(input=dist_mat, beta=1, mat1=_seq, mat2=_seq.t(), alpha=-2)
    dist_mat = dist_mat[:, pick_idxs].min(1, keepdim=True)[0]
    
    scale = 0.1 if _seq.size(1) == 2048 else 1.0
    reward_rep = torch.exp(-dist_mat.mean() * scale)

    # SAC Dense Distribution: Do NOT divide by num_picks
    total_val = (reward_div * div_coff + reward_rep * rep_coff)
    step_reward[pick_idxs] = total_val 
    
    return step_reward

def compute_reward_det_coff(seq, actions, det_scores, det_class,
                            ignore_far_sim=True, temp_dist_thre=20,
                            div_coff=1.0, rep_coff=1.0, det_coff=1.0):
    """
    Detection-based reward optimized for SAC.
    Note: Coefficients adjusted for SAC scale (1.0 vs 50.0).
    """
    _seq = seq.detach().squeeze()
    _actions = actions.detach().squeeze()
    n = _seq.size(0)
    step_reward = torch.zeros(n, device=device)

    pick_idxs = _actions.nonzero().squeeze()
    if pick_idxs.ndimension() == 0 and pick_idxs.numel() > 0:
        pick_idxs = pick_idxs.unsqueeze(0)
    num_picks = len(pick_idxs) if pick_idxs.numel() > 0 else 0

    if num_picks == 0:
        return step_reward

    # Global Diversity & Rep
    normed_seq = _seq / (_seq.norm(p=2, dim=1, keepdim=True) + 1e-8)
    dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
    if num_picks > 1:
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.view(-1, 1).expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))
    else:
        reward_div = torch.tensor(0., device=device)

    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat = torch.addmm(input=dist_mat, beta=1, mat1=_seq, mat2=_seq.t(), alpha=-2)
    reward_rep = torch.exp(-dist_mat[:, pick_idxs].min(1, keepdim=True)[0].mean())

    # Detection Logic
    det_class = torch.from_numpy(det_class.astype(int)).to(device)
    det_scores = torch.from_numpy(det_scores).to(device)
    sp_det_score = torch.zeros(n, device=device)

    # Class 3 is assumed to be "Background/Noise" based on your PPO logic
    for i in range(n):
        cls_idx = det_class[i][0]
        if cls_idx != 3:
            sp_det_score[i] = det_scores[i][cls_idx]
        else:
            sp_det_score[i] = -det_scores[i][3]

    # Combine for SAC: 
    # Global metrics are shared across picked frames, detection is frame-specific.
    # No division by num_picks.
    base_share = (reward_div * div_coff + reward_rep * rep_coff)
    step_reward[pick_idxs] = base_share + (sp_det_score[pick_idxs] * det_coff)

    return step_reward