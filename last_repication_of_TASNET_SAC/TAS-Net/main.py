from __future__ import print_function
import torch
import torch.optim as optim
import os.path as osp
import argparse
import os
import torch.nn as nn
import h5py
from torch.optim import lr_scheduler
import torch.optim as optim
from models import Actor, Critic, TransformerActor, MultiScaleActor # Updated SAC Models
from utils import weights_init, read_json
from rewards import compute_reward
import numpy as np
import random
from scipy.io import savemat
from Graph_Net import ClassifierGNN
from evaluate import evaluate
import math
import collections

# Parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--training', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--subject_id', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--hid_dim', type=int, default=256)
parser.add_argument('--deep_features', type=str, default='./features/source_h5_file.h5')
parser.add_argument('--save_path', type=str, default='./checkpoints')
parser.add_argument('--fragment_length', type=int, default=8)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--variant', type=str, default='sac', choices=['sac', 'sac_transformer', 'sac_multiscale'])
parser.add_argument('--n_feature', type=int, default=192)
parser.add_argument('--edge_features', type=int, default=32)
parser.add_argument('--num_fragment', type=int, default=5)
parser.add_argument('--reward_function', type=str, default='sac')
parser.add_argument('--start_steps', type=int, default=1000, help='Replay warmup steps')
# SAC Specific Hyperparameters
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
parser.add_argument('--alpha', type=float, default=0.2, help='Entropy temperature')
parser.add_argument('--lr_alpha', type=float, default=3e-4, help='Alpha learning rate')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--buffer_size', type=int, default=100000)

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- REPLAY BUFFER FOR SAC ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        from torch.nn.utils.rnn import pad_sequence
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (pad_sequence(state, batch_first=True), 
                pad_sequence(action, batch_first=True), 
                pad_sequence(reward, batch_first=True), 
                pad_sequence(next_state, batch_first=True), 
                torch.stack(done))

    def __len__(self):
        return len(self.buffer)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

if __name__ == '__main__':
    if args.training:
        datasets = h5py.File(args.deep_features, 'r')
        all_keys = list(datasets.keys())
        test_keys = all_keys[15 * args.subject_id:15 * (args.subject_id + 1)]
        train_keys = [k for k in all_keys if k not in test_keys]

        # 1. Models Initialization
        Network1 = ClassifierGNN(in_features=args.n_feature, edge_features=args.edge_features, out_features=args.n_feature, device=DEVICE).to(DEVICE)
        
        if args.variant == 'sac':
            actor = Actor(in_dim=args.n_feature, hid_dim=args.hid_dim).to(DEVICE)
        elif args.variant == 'sac_transformer':
            actor = TransformerActor(in_dim=args.n_feature, hid_dim=args.hid_dim).to(DEVICE)
        elif args.variant == 'sac_multiscale':
            actor = MultiScaleActor(in_dim=args.n_feature, hid_dim=args.hid_dim).to(DEVICE)
            
        critic = Critic(in_dim=args.n_feature, hid_dim=args.hid_dim).to(DEVICE)
        critic_target = Critic(in_dim=args.n_feature, hid_dim=args.hid_dim).to(DEVICE)
        critic_target.load_state_dict(critic.state_dict())

        actor.apply(weights_init)
        critic.apply(weights_init)

        # 2. Optimizers
        lr_actor = 3e-5 if args.variant == 'sac_transformer' else args.lr
        lr_critic = 3e-5 if args.variant == 'sac_transformer' else args.lr
        lr_alpha_val = 1e-5
        
        optimizer_gnn = optim.Adam(Network1.parameters(), lr=args.lr)
        optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
        optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
        
        
        # Automatic Entropy Tuning (Fixed for Discrete SAC)
        target_entropy = 0.98 * math.log(2.0)
        log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        alpha_optim = optim.Adam([log_alpha], lr=lr_alpha_val)

        replay_buffer = ReplayBuffer(args.buffer_size)
        best_recall = 0.0
        global_step = 0

        for epoch in range(args.epochs):
            Network1.train()
            actor.train()
            critic.train()
            
            if args.variant == 'sac_transformer':
                for param in actor.transformer.parameters():
                    param.requires_grad = (epoch >= 5)

            np.random.shuffle(train_keys)
            epoch_rewards = []
            epoch_actor_loss = []
            epoch_critic_loss = []
            epoch_q_val = []
            epoch_entropy = []

            for key in train_keys:
                # --- Load & GNN Process ---
                seq_raw = torch.from_numpy(datasets[key]['features'][...]).float().to(DEVICE)
                with torch.no_grad():
                    local_graphs = []
                    for n in range(math.ceil(seq_raw.shape[0] / (args.fragment_length * 2))):
                        chunk = seq_raw[(args.fragment_length * 2) * n : (args.fragment_length * 2) * (n+1)]
                        if chunk.shape[0] > 1:
                            sub_g, _ = Network1(chunk)
                            local_graphs.append(sub_g)
                        else:
                            local_graphs.append(chunk)
                    local_graphs = torch.cat(local_graphs, dim=0)
                
                state = (seq_raw + local_graphs).unsqueeze(0) # (1, T, D)

                # --- 3. Interaction (Collect Experience) ---
                if global_step < args.start_steps:
                    actions = torch.distributions.Bernoulli(torch.full((1, state.size(1), 1), 0.5)).sample().to(DEVICE)
                else:
                    logits = actor(state)
                    probs = torch.sigmoid(logits)
                    dist = torch.distributions.Bernoulli(probs)
                    actions = dist.sample()
                
                rewards = compute_reward(state, actions)
                
                # Push to buffer as a whole sequence transition
                # In TAS-Net, next_state is essentially terminal for a summary 
                replay_buffer.push(state.squeeze(0), actions.squeeze(0), rewards, state.squeeze(0), torch.tensor([1.0]))
                global_step += 1

                # --- 4. SAC Update Loop ---
                if len(replay_buffer) > args.batch_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(args.batch_size)
                    b_s, b_a, b_r, b_ns, b_d = b_s.to(DEVICE), b_a.to(DEVICE), b_r.to(DEVICE), b_ns.to(DEVICE), b_d.to(DEVICE)
                    
                    # Current Alpha
                    alpha = log_alpha.exp()

                    # CRITIC UPDATE
                    with torch.no_grad():
                        next_logits = actor(b_ns)
                        p1_next = torch.sigmoid(next_logits)
                        p0_next = 1.0 - p1_next
                        next_probs = torch.cat([p0_next, p1_next], dim=-1)
                        
                        q1_next, q2_next = critic_target(b_ns)
                        min_q_next = torch.min(q1_next, q2_next)
                        
                        log_probs_next = torch.log(next_probs + 1e-8)
                        # Exact expected V(s) over discrete actions
                        next_v = (next_probs * (min_q_next - alpha * log_probs_next)).sum(dim=-1)
                        target_q = b_r + (1 - b_d) * args.gamma * next_v

                    curr_q1, curr_q2 = critic(b_s)
                    curr_q1_a = curr_q1.gather(-1, b_a.long())
                    curr_q2_a = curr_q2.gather(-1, b_a.long())
                    
                    critic_loss = nn.MSELoss()(curr_q1_a, target_q.unsqueeze(-1)) + \
                                  nn.MSELoss()(curr_q2_a, target_q.unsqueeze(-1))

                    optimizer_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    optimizer_critic.step()

                    # ACTOR UPDATE
                    curr_logits = actor(b_s)
                    p1_curr = torch.sigmoid(curr_logits)
                    p0_curr = 1.0 - p1_curr
                    curr_probs = torch.cat([p0_curr, p1_curr], dim=-1)
                    log_probs_curr = torch.log(curr_probs + 1e-8)
                    
                    q1_new, q2_new = critic(b_s)
                    min_q_new = torch.min(q1_new, q2_new)
                    
                    entropy_val = - (curr_probs * log_probs_curr).sum(dim=-1)
                    # Policy objective: minimize exact expected KL divergence
                    actor_loss = (curr_probs * (alpha * log_probs_curr - min_q_new)).sum(dim=-1).mean()

                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optimizer_actor.step()

                    # ALPHA UPDATE (Entropy Tuning)
                    alpha_loss = -(log_alpha * (-entropy_val.detach() + target_entropy)).mean()
                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()
                    
                    # Clamp alpha to prevent total collapse of exploration
                    with torch.no_grad():
                        log_alpha.clamp_(min=math.log(0.05))

                    soft_update(critic_target, critic, args.tau)
                    
                    epoch_actor_loss.append(actor_loss.item())
                    epoch_critic_loss.append(critic_loss.item())
                    epoch_q_val.append(min_q_new.mean().item())
                    epoch_entropy.append(entropy_val.mean().item())

                epoch_rewards.append(rewards.mean().item())

            # --- Eval & Save (Keep your existing logic) ---
            avg_reward = np.mean(epoch_rewards)
            a_l = np.mean(epoch_actor_loss) if epoch_actor_loss else 0.0
            c_l = np.mean(epoch_critic_loss) if epoch_critic_loss else 0.0
            q_v = np.mean(epoch_q_val) if epoch_q_val else 0.0
            ent = np.mean(epoch_entropy) if epoch_entropy else 0.0
            print(f"Epoch {epoch+1} | Reward: {avg_reward:.6f} | Alpha: {log_alpha.exp().item():.4f}")
            print(f"   [Debug] A-Loss: {a_l:.4f} | C-Loss: {c_l:.4f} | Q-val: {q_v:.4f} | Ent: {ent:.4f}")
            
            Recall = evaluate(args, Network1, actor, datasets, test_keys)
            # Restore train mode after evaluate() sets models to eval()
            Network1.train()
            actor.train()
            print(f"Recall: {Recall:.6f}")
            if Recall > best_recall:
                best_recall = Recall
                torch.save(Network1.state_dict(), osp.join(args.save_path, f'best_gnn_subj{args.subject_id}.pth'))
                torch.save(actor.state_dict(), osp.join(args.save_path, f'best_actor_subj{args.subject_id}.pth'))
                print("🔥 Best SAC model saved!")

    else:
        # --- TESTING MODE ---
        # Note: In Testing, use actor.get_action_probs() for inference
        print("Running Inference with SAC Actor...")
        # [Existing testing logic using actor instead of Network2]
        datasets = h5py.File(args.deep_features, 'r')
        all_keys = list(datasets.keys())
        test_keys = all_keys[15 * args.subject_id:15 * (args.subject_id + 1)]

        print("# test videos {}.".format(len(test_keys)))
        model1 = ClassifierGNN(in_features=args.n_feature, edge_features=args.edge_features, out_features=args.n_feature, device=DEVICE).to(DEVICE)
        
        if args.variant == 'sac':
            model2 = Actor(in_dim=args.n_feature, hid_dim=args.hid_dim).to(DEVICE)
        elif args.variant == 'sac_transformer':
            model2 = TransformerActor(in_dim=args.n_feature, hid_dim=args.hid_dim).to(DEVICE)
        elif args.variant == 'sac_multiscale':
            model2 = MultiScaleActor(in_dim=args.n_feature, hid_dim=args.hid_dim).to(DEVICE)

        checkpoint_path1 = osp.join(args.save_path, f'best_gnn_subj{args.subject_id}.pth')
        checkpoint_path2 = osp.join(args.save_path, f'best_actor_subj{args.subject_id}.pth')
        
        model1.load_state_dict(torch.load(checkpoint_path1, map_location=DEVICE, weights_only=False))
        model2.load_state_dict(torch.load(checkpoint_path2, map_location=DEVICE, weights_only=False))

        with torch.no_grad():
            model1.eval()
            model2.eval()
            out_path = os.path.join(args.save_path, 'result_output')
            if not os.path.exists(out_path): os.makedirs(out_path)
            
            save_idx = open(os.path.join(out_path, f'log_subject{args.subject_id}_{args.reward_function}.txt'), 'w')
            
            all_features = None
            all_labels = None
            num_segments_trial = []

            local_labels = [
                [[13, 22], [204, 218], [219, 235]], [[50, 65], [132, 150], [164, 187], [206, 226]],
                [[14, 27], [66, 80], [135, 149], [150, 165], [186, 206]], [[4, 24], [26, 45], [95, 121], [131, 136], [166, 183], [202, 212]],
                [[15, 35], [35, 50], [135, 150]], [[10, 19], [40, 49], [63, 74], [91, 103], [120, 129], [165, 181]],
                [[23, 40], [61, 75], [152, 165], [180, 195], [200, 212]], [], [[55, 70], [128, 143], [165, 180], [215, 235]],
                [[14, 34], [58, 83], [98, 108], [141, 151]], [], [[45, 63], [76, 91], [148, 159], [188, 204], [209, 219], [229, 240]],
                [[21, 31], [92, 103], [119, 129], [224, 240]], [[49, 60], [138, 150], [162, 174], [195, 210]],
                [[63, 80], [97, 113], [120, 134], [165, 180], [184, 205]]
            ]

            for key_idx, key in enumerate(test_keys):
                seq = torch.from_numpy(datasets[key]['features'][...]).float().to(DEVICE)  # ensure float32
                gt = datasets[key]['labels'][...]
                label_idx = key_idx
                local_label = local_labels[label_idx]

                sub_graphs = []
                for n in range(math.ceil(seq.shape[0] / (args.fragment_length * 2))):
                    chunk = seq[(args.fragment_length * 2) * n : (args.fragment_length * 2) * (n + 1), :]
                    if chunk.shape[0] <= 1:
                        sub_graphs.append(chunk)  # skip GNN for single-frame chunks
                    else:
                        sg, _ = model1(chunk)
                        sub_graphs.append(sg)

                sub_graphs = torch.cat(sub_graphs, dim=0)
                seq_graph = torch.add(seq, sub_graphs).unsqueeze(0)

                sig_probs = model2.get_action_probs(seq_graph)
                # Ensure probs_importance is always 1-D (handles single-frame sequences)
                probs_importance = sig_probs.data.cpu().squeeze().numpy()
                if probs_importance.ndim == 0:
                    probs_importance = probs_importance.reshape(1)

                # Clamp limits to actual sequence length to prevent IndexError
                limits = min(args.num_fragment, len(probs_importance))
                order = np.argsort(probs_importance)[::-1]

                all_fragment = []
                n_t = 0
                if label_idx != 7 and label_idx != 10:
                    for j in range(len(local_label)):
                        for i in range(limits):
                            gt_left_idx = local_label[j][0]
                            gt_right_idx = local_label[j][1]
                            idx = order[i] + args.fragment_length
                            left_idx = idx - probs_importance[idx - args.fragment_length] * args.fragment_length
                            left_int_idx = int(np.ceil(left_idx))
                            right_idx = idx + probs_importance[idx - args.fragment_length] * args.fragment_length
                            right_int_idx = int(np.floor(right_idx))
                            
                            if left_int_idx - args.fragment_length >= gt_right_idx or right_int_idx - args.fragment_length <= gt_left_idx:
                                tIOU = 0.
                            else:
                                idx_set = np.hstack((gt_left_idx, gt_right_idx))
                                idx_set = np.hstack((idx_set, left_int_idx - args.fragment_length))
                                idx_set = np.hstack((idx_set, right_int_idx - args.fragment_length))
                                idx_set = np.sort(idx_set)
                                tIOU = (idx_set[2] - idx_set[1]) / (idx_set[3] - idx_set[0])
                            
                            if tIOU >= 0.5:
                                n_t += 1
                                break
                    local_recall = n_t / len(local_label)
                    log_str1 = 'i_th trial %.0f\tRecall %.02f' % (label_idx, local_recall)
                    save_idx.write(log_str1 + '\n')
                    save_idx.flush()

                for i in range(limits):
                    idx = order[i] + args.fragment_length
                    left_idx = idx - probs_importance[idx - args.fragment_length] * args.fragment_length
                    left_int_idx = max(0, int(np.ceil(left_idx)) - args.fragment_length)
                    right_idx = idx + probs_importance[idx - args.fragment_length] * args.fragment_length
                    right_int_idx = min(seq.shape[0], int(np.floor(right_idx)) - args.fragment_length)
                    
                    if right_int_idx > left_int_idx:
                        one_fragment = seq[left_int_idx:right_int_idx, ]
                        all_fragment.append(one_fragment)

                    log_str0 = 'i_th fragment %.0f\tleft_idx %.0f\tright_idx %.0f' % (i, left_int_idx, right_int_idx)
                    save_idx.write(log_str0 + '\n')
                    save_idx.flush()

                if len(all_fragment) > 0:
                    all_fragment = torch.vstack(all_fragment)
                    
                    # --- FIXED: Handle 0-d and 1-d labels safely ---
                    if isinstance(gt, np.ndarray):
                        gt_val = gt.item() if gt.ndim == 0 else gt[0]
                    else:
                        gt_val = gt
                    
                    labels = torch.full((all_fragment.shape[0], 1), gt_val).to(DEVICE)
                    num_segments_trial.append(all_fragment.shape[0])

                    if all_features is not None:
                        all_features = torch.cat((all_features, all_fragment), dim=0)
                        all_labels = torch.cat((all_labels, labels), dim=0)
                    else:
                        all_features = all_fragment
                        all_labels = labels

            # Final check to see if we actually collected any features
            if all_features is not None:
                all_features = all_features.cpu().data.numpy()
                all_labels = all_labels.cpu().data.numpy()
                
                print(f"Final shape: Features {all_features.shape}, Labels {all_labels.shape}")
                
                mat_file = os.path.join(out_path, f'TAS_subject{args.subject_id}_{args.reward_function}_{args.num_fragment}.mat')
                savemat(mat_file, {'feature': all_features, 'label': all_labels})
            else:
                print("Warning: No fragments were extracted during testing.")

            save_idx.close() 

            # Calculate and print the average Recall across all test videos
            all_recalls = []
            log_filepath = os.path.join(out_path, f'log_subject{args.subject_id}_{args.reward_function}.txt')

            if os.path.exists(log_filepath):
                with open(log_filepath, 'r') as f:
                    for line in f:
                        if 'Recall' in line:
                            try:
                                # Log format: 'i_th trial X\tRecall Y.YY'
                                parts = line.strip().split('Recall')
                                recall_val = float(parts[-1].strip())
                                all_recalls.append(recall_val)
                            except (ValueError, IndexError):
                                continue

            print("\n" + "="*40)
            print(f"TESTING SUMMARY (Subject {args.subject_id})")
            if all_recalls:
                mean_recall = np.mean(all_recalls)
                print(f"Mean Recall @ IoU>=0.5 : {mean_recall:.4f}  (over {len(all_recalls)} valid trials)")
            else:
                print("No recall values collected — check that test_keys contain valid labelled trials.")
            print(f"Log saved to           : {log_filepath}")
            print("="*40 + "\n")
            
        # This is the end of the torch.no_grad() block
        datasets.close()
