import torch
import random
from utils import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def compute_reward(seq, actions, ignore_far_sim=False, temp_dist_thre=20):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze()
    pick_idxs = torch.nonzero(pick_idxs, as_tuple=False)
    pick_idxs = pick_idxs.squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        reward = reward.to(device)
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_sim = torch.tensor(0.)
        reward_sim = reward_sim.to(device)
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        sim_mat = torch. matmul(normed_seq, normed_seq.t())  # similarity matrix [Eq.4]
        sim_submat = sim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            sim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_sim = sim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat = torch.addmm(input=dist_mat, beta=1, mat1=_seq, mat2=_seq.t(), alpha=-2)
    dist_mat = dist_mat[:, pick_idxs]
    if len(dist_mat.shape) == 1:
        dist_mat = dist_mat.unsqueeze(dim=1)
    dist_mat = torch.min(dist_mat, 1, keepdim=True)
    dist_mat = dist_mat[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    reward = (reward_sim + reward_rep) * 0.5
    return reward

def compute_reward_coff(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False,
                        div_coff=0.5, rep_coff=0.5):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu:
            reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1 - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:, pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]

    if _seq.size(1) == 2048:
        reward_rep = torch.exp(-dist_mat.mean()*0.1)
    else:
        reward_rep = torch.exp(-dist_mat.mean())

    # combine the two rewards
    reward = (reward_div * div_coff + reward_rep * rep_coff)
    # print("num_picks {} reward_div {}\t  reward_rep {}\t".format(num_picks, reward_div, reward_rep))
    return reward


def compute_reward_det_coff(seq, actions, det_scores, det_class,  episode,
                            ignore_far_sim=True, temp_dist_thre=20, use_gpu=False,
                            div_coff=50.0, rep_coff=50.0, det_coff =50.0):
    """
    Compute detection reward, diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU

    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu:
            reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:, pick_idxs]

    dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    det_class = torch.from_numpy(det_class.astype(int))
    det_scores = torch.from_numpy(det_scores)
    sp_det_score = torch.zeros(n) # the score of standard plane detection

    for i in range(n):
        det_cls = det_class[i]
        if det_cls[0] != 3:
            sp_det_score[i] = det_scores[i][det_cls]
        else:
            sp_det_score[i] = -det_scores[i][3]

    pick_scores = sp_det_score[pick_idxs]
    reward_det = torch.sum(pick_scores)/num_picks # (norm_summ_scores)

    # combine the three rewards
    reward = reward_div * div_coff + reward_rep * rep_coff + reward_det  * det_coff
    # reward = (reward_div + reward_rep) * 0.25 + reward_det *0.5
    return reward


# =============================================================================
# NEW REWARD FUNCTIONS  (R1 – R4)
# Original compute_reward / compute_reward_coff / compute_reward_det_coff above
# are COMPLETELY UNTOUCHED.
# =============================================================================

REWARD_DESCRIPTIONS = {
    'original': 'Original TAS-Net Diversity + Representativeness Reward',
    'r1':       'Coverage + Diversity + Low Redundancy Reward',
    'r2':       'Boundary / Transition-Aware Reward',
    'r3':       'Compactness + Separation Reward',
    'r4':       'Temporal Smoothness + Contiguity Reward',
}


def _get_picks(actions):
    """Return (pick_idxs tensor, num_picks int) from binary action tensor."""
    pick_idxs = torch.nonzero(actions.detach().squeeze(), as_tuple=False).squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    return pick_idxs, num_picks


def reward_r1(seq, actions, num_fragment=10):
    """
    R1 — Coverage + Diversity + Low Redundancy Reward
    --------------------------------------------------
    R1 = 1.0*C + 0.7*D - 0.5*U - 0.3*P

    C  : coverage  = exp(-mean_i min_{j in S} ||h_i - h_j||_2)
    D  : pairwise L2 distance mean among selected frames
    U  : mean cosine similarity among selected frames (redundancy)
    P  : budget penalty = (rho - rho_target)^2
    """
    _seq     = seq.detach().squeeze()          # (T, dim)
    _actions = actions.detach().squeeze()
    T        = _seq.size(0)

    pick_idxs, num_picks = _get_picks(_actions)

    if num_picks == 0:
        return torch.tensor(-1.0).to(device)

    rho        = num_picks / T
    rho_target = num_fragment / T
    P          = (rho - rho_target) ** 2

    selected = _seq[pick_idxs].view(-1, _seq.size(1))  # Force 2D (|S|, dim)

    # --- Coverage: how well S covers the full sequence ---
    dist_to_S = torch.cdist(_seq.unsqueeze(0), selected.unsqueeze(0)).squeeze(0)  # (T, |S|)
    C = torch.exp(-dist_to_S.min(dim=1).values.mean())

    # --- Diversity: mean pairwise L2 among selected ---
    if num_picks == 1:
        D = torch.tensor(0.0).to(device)
    else:
        pw   = torch.cdist(selected.unsqueeze(0), selected.unsqueeze(0)).squeeze(0)
        mask = ~torch.eye(num_picks, dtype=torch.bool, device=device)
        D    = pw[mask].mean()

    # --- Redundancy: mean cosine similarity among selected ---
    if num_picks == 1:
        U = torch.tensor(0.0).to(device)
    else:
        normed = selected / (selected.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sim    = torch.matmul(normed, normed.t())
        mask   = ~torch.eye(num_picks, dtype=torch.bool, device=device)
        U      = sim[mask].mean()

    reward = 1.0 * C + 0.7 * D - 0.5 * U - 0.3 * P
    return torch.clamp(reward, -5.0, 5.0)


def reward_r2(seq, actions, num_fragment=10):
    """
    R2 — Boundary / Transition-Aware Reward
    ----------------------------------------
    R2 = 1.0*B + 0.5*C - 0.5*U - 0.3*P

    B  : mean boundary score of selected frames
         B_t = (||h_t - h_{t-1}|| + ||h_{t+1} - h_t||) / 2
    C  : coverage (same as R1)
    U  : redundancy (same as R1)
    P  : budget penalty
    """
    _seq     = seq.detach().squeeze()          # (T, dim)
    _actions = actions.detach().squeeze()
    T        = _seq.size(0)

    pick_idxs, num_picks = _get_picks(_actions)

    if num_picks == 0:
        return torch.tensor(-1.0).to(device)

    rho        = num_picks / T
    rho_target = num_fragment / T
    P          = (rho - rho_target) ** 2

    selected = _seq[pick_idxs].view(-1, _seq.size(1))

    # --- Boundary score per frame (central finite diff, clamped at edges) ---
    left  = torch.cat([_seq[:1], _seq[:-1]], dim=0)   # h_{t-1}
    right = torch.cat([_seq[1:], _seq[-1:]], dim=0)   # h_{t+1}
    boundary_all = ((_seq - left).norm(p=2, dim=1) +
                    (right - _seq).norm(p=2, dim=1)) / 2.0   # (T,)
    B = boundary_all[pick_idxs].mean()

    # --- Coverage ---
    dist_to_S = torch.cdist(_seq.unsqueeze(0), selected.unsqueeze(0)).squeeze(0)
    C = torch.exp(-dist_to_S.min(dim=1).values.mean())

    # --- Redundancy ---
    if num_picks == 1:
        U = torch.tensor(0.0).to(device)
    else:
        normed = selected / (selected.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sim    = torch.matmul(normed, normed.t())
        mask   = ~torch.eye(num_picks, dtype=torch.bool, device=device)
        U      = sim[mask].mean()

    reward = 1.0 * B + 0.5 * C - 0.5 * U - 0.3 * P
    return torch.clamp(reward, -5.0, 5.0)


def reward_r3(seq, actions, num_fragment=10):
    """
    R3 — Compactness + Separation Reward
    --------------------------------------
    R3 = 1.0*Sep - 0.7*Comp - 0.3*U - 0.3*P

    Sep  : ||mu_S - mu_U||_2   (selected vs unselected centroid distance)
    Comp : mean distance of selected frames to their centroid
    U    : redundancy
    P    : budget penalty
    """
    _seq     = seq.detach().squeeze()          # (T, dim)
    _actions = actions.detach().squeeze()
    T        = _seq.size(0)

    pick_idxs, num_picks = _get_picks(_actions)

    if num_picks == 0:
        return torch.tensor(-1.0).to(device)

    rho        = num_picks / T
    rho_target = num_fragment / T
    P          = (rho - rho_target) ** 2

    selected = _seq[pick_idxs].view(-1, _seq.size(1))

    # Unselected frames
    mask_selected = torch.zeros(T, dtype=torch.bool, device=device)
    mask_selected[pick_idxs] = True
    unselected = _seq[~mask_selected].view(-1, _seq.size(1))

    mu_S = selected.mean(dim=0)

    # --- Compactness ---
    Comp = (selected - mu_S).norm(p=2, dim=1).mean()

    # --- Separation ---
    if unselected.size(0) == 0:
        Sep = torch.tensor(0.0).to(device)
    else:
        mu_U = unselected.mean(dim=0)
        Sep  = (mu_S - mu_U).norm(p=2)

    # --- Redundancy ---
    if num_picks == 1:
        U = torch.tensor(0.0).to(device)
    else:
        normed = selected / (selected.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sim    = torch.matmul(normed, normed.t())
        mask   = ~torch.eye(num_picks, dtype=torch.bool, device=device)
        U      = sim[mask].mean()

    reward = 1.0 * Sep - 0.7 * Comp - 0.3 * U - 0.3 * P
    return torch.clamp(reward, -5.0, 5.0)


def reward_r4(seq, actions, num_fragment=10):
    """
    R4 — Temporal Smoothness + Contiguity Reward
    ----------------------------------------------
    R4 = 1.0*Contig + 0.5*C - 0.5*Iso - 0.3*U - 0.3*P

    Contig : avg selected-run length / T
    Iso    : fraction of isolated selected frames
    C      : coverage
    U      : redundancy
    P      : budget penalty
    """
    _seq     = seq.detach().squeeze()          # (T, dim)
    _actions = actions.detach().squeeze()
    T        = _seq.size(0)

    pick_idxs, num_picks = _get_picks(_actions)

    if num_picks == 0:
        return torch.tensor(-1.0).to(device)

    rho        = num_picks / T
    rho_target = num_fragment / T
    P          = (rho - rho_target) ** 2

    selected = _seq[pick_idxs].view(-1, _seq.size(1))

    # --- Build binary selection mask ---
    sel_mask = torch.zeros(T, dtype=torch.bool, device=device)
    sel_mask[pick_idxs] = True
    sel_np = sel_mask.cpu().numpy().astype(int)

    # --- Contiguity: avg run length of selected frames / T ---
    runs, run = [], 0
    for v in sel_np:
        if v == 1:
            run += 1
        else:
            if run > 0:
                runs.append(run)
                run = 0
    if run > 0:
        runs.append(run)
    Contig = torch.tensor(
        (sum(runs) / len(runs) / T) if runs else 0.0,
        dtype=torch.float32
    ).to(device)

    # --- Isolation: fraction of selected frames with no neighbour selected ---
    iso_count = 0
    for i, v in enumerate(sel_np):
        if v == 1:
            left_ok  = (i > 0     and sel_np[i - 1] == 1)
            right_ok = (i < T - 1 and sel_np[i + 1] == 1)
            if not left_ok and not right_ok:
                iso_count += 1
    Iso = torch.tensor(iso_count / T, dtype=torch.float32).to(device)

    # --- Coverage ---
    dist_to_S = torch.cdist(_seq.unsqueeze(0), selected.unsqueeze(0)).squeeze(0)
    C = torch.exp(-dist_to_S.min(dim=1).values.mean())

    # --- Redundancy ---
    if num_picks == 1:
        U = torch.tensor(0.0).to(device)
    else:
        normed = selected / (selected.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sim    = torch.matmul(normed, normed.t())
        mask   = ~torch.eye(num_picks, dtype=torch.bool, device=device)
        U      = sim[mask].mean()

    reward = 1.0 * Contig + 0.5 * C - 0.5 * Iso - 0.3 * U - 0.3 * P
    return torch.clamp(reward, -5.0, 5.0)


# =============================================================================
# DISPATCHER  — single entry point used by main.py
# =============================================================================

def dispatch_reward(reward_type, seq, actions, num_fragment=10):
    """
    Select and compute the reward function by name.

    Args:
        reward_type  : str  — 'original' | 'r1' | 'r2' | 'r3' | 'r4'
        seq          : Tensor (1, T, dim) — graph-enhanced features
        actions      : Tensor (1, T, 1)  — sampled binary actions
        num_fragment : int  — target number of fragments (from args.num_fragment)

    Returns:
        scalar reward Tensor
    """
    if reward_type == 'original':
        return compute_reward(seq, actions)
    elif reward_type == 'r1':
        return reward_r1(seq, actions, num_fragment)
    elif reward_type == 'r2':
        return reward_r2(seq, actions, num_fragment)
    elif reward_type == 'r3':
        return reward_r3(seq, actions, num_fragment)
    elif reward_type == 'r4':
        return reward_r4(seq, actions, num_fragment)
    else:
        raise ValueError(
            f"Unknown reward_type '{reward_type}'. "
            f"Choose from: {list(REWARD_DESCRIPTIONS.keys())}"
        )
