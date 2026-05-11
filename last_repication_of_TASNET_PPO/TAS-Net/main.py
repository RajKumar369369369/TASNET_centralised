from __future__ import print_function
from torch import optim
import torch
import os.path as osp
import time
import argparse
import datetime
import os
import torch.nn as nn
import h5py
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from models import DSN
from utils import weights_init, save_checkpoint, inv_lr_scheduler, read_json
from rewards import compute_reward_det_coff, compute_reward_coff, compute_reward
import numpy as np
import random
from scipy.io import savemat
from Graph_Net import ClassifierGNN
from evaluate import evaluate
import math

# Parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--training', action='store_true', default=False, help='Training or Validate.')
parser.add_argument('--seed', type=int, default=27, help='Random seed')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--subject_id', type=int, default=0, help="subject index (default: 0)")
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--edge_features', type=int, default=32, help='graph edge features dimension.')
parser.add_argument('--n_feature', type=int, default=192, help='Number of hidden units.')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hid_dim', type=int, default=256, help='hidden unit dimension of DSN (default: 256).')
parser.add_argument('--deep_features', type=str, default='./features/SEED/session_1/source_h5_file.h5', help='output directory')
parser.add_argument('--save_path', type=str, default='./checkpoints', help='output directory')
parser.add_argument('--fragment_length', type=int, default=8, help='Left or Right Maximum offset.')
parser.add_argument('--num_fragment', type=int, default=10, help='for eval emotion localization.')
parser.add_argument('--reward_function', type=str, default='R1_R2', help='sim:R1, rep:R2 or mix:R1_R2.')
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use.")

# PPO Hyperparameters
parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clipping epsilon.')
parser.add_argument('--k_epochs', type=int, default=10, help='Number of optimization epochs per video.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter.')

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compute_gae(rewards, values, next_value, gamma, lam):
    """Calculates Advantages and Returns using GAE with Professor's Normalization Fix."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    # Ensure inputs are treated as sequences [T]
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] + gamma * next_value - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]

        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    
    # PROFESSOR'S FIX: Advantage Normalization
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
    return advantages, returns

if __name__ == '__main__':
    if args.training:

         datasets = h5py.File(args.deep_features, 'r')
         all_keys = list(datasets.keys())

         # Subject-wise split
         test_keys = all_keys[15 * args.subject_id:15 * (args.subject_id + 1)]
         train_keys = [k for k in all_keys if k not in test_keys]

         print(f"# total {len(all_keys)} | train {len(train_keys)} | test {len(test_keys)}")

         # Models
         Network1 = ClassifierGNN(
            in_features=args.n_feature,
            edge_features=args.edge_features,
            out_features=args.n_feature,
            device=DEVICE
         ).to(DEVICE)

         Network2 = DSN(
            in_dim=args.n_feature,
            hid_dim=args.hid_dim,
            num_layers=1,
            cell='gru'
         ).to(DEVICE)

         Network2.apply(weights_init)

         optimizer = optim.Adam(
            list(Network1.parameters()) + list(Network2.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
         )

         scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

         print("=====> Start PPO Training <=====")

         best_recall = 0.0

         for epoch in range(args.epochs):

             np.random.shuffle(train_keys)
             epoch_rewards = []

             for key in train_keys:

                # =========================
                # 1. Load data
                # =========================
                 seq_raw = torch.from_numpy(
                    datasets[key]['features'][...]
                 ).float().to(DEVICE)

                # =========================
                # 2. GNN Feature Extraction
                # =========================
                 with torch.no_grad():
                     local_graphs = []

                     for n in range(math.ceil(seq_raw.shape[0] / (args.fragment_length * 2))):
                         chunk = seq_raw[
                             (args.fragment_length * 2) * n:
                             (args.fragment_length * 2) * (n + 1)
                         ]

                         if chunk.shape[0] > 1:
                             sub_g, _ = Network1(chunk)
                             local_graphs.append(sub_g)
                         else:
                             local_graphs.append(chunk)

                     local_graphs = torch.cat(local_graphs, dim=0)

                 seq_graph = (seq_raw + local_graphs).unsqueeze(0)  # (1, T, D)

                # =========================
                # 3. Forward pass
                # =========================
                 probs, values = Network2(seq_graph)

                 probs = probs.squeeze(0)       # (T, 1)
                 values = values.squeeze(0)     # (T, 1)

                 dist = torch.distributions.Bernoulli(probs)

                 actions = dist.sample()
                 log_probs_old = dist.log_prob(actions).detach()

                # =========================
                # 4. DENSE reward (Professor's Update)
                # =========================
                 # Updated compute_reward returns a tensor [T, 1] instead of scalar
                 rewards = compute_reward(seq_graph, actions)  

                # =========================
                # 5. GAE
                # =========================
                 advantages, returns = compute_gae(
                    rewards.squeeze(-1),
                    values.detach().squeeze(-1),
                    next_value=0,
                    gamma=args.gamma,
                    lam=args.gae_lambda
                 )

                # =========================
                # 6. PPO UPDATE
                # =========================
                 for _ in range(4):

                     curr_probs, curr_values = Network2(seq_graph)

                     curr_probs = curr_probs.squeeze(0)
                     curr_values = curr_values.squeeze(0)

                     curr_dist = torch.distributions.Bernoulli(curr_probs)

                     curr_log_probs = curr_dist.log_prob(actions)
                     entropy = curr_dist.entropy().mean()

                     ratio = torch.exp(curr_log_probs - log_probs_old)

                     surr1 = ratio * advantages.unsqueeze(-1)
                     surr2 = torch.clamp(
                        ratio,
                        1 - args.eps_clip,
                        1 + args.eps_clip
                     ) * advantages.unsqueeze(-1)

                     actor_loss = -torch.min(surr1, surr2).mean()

                     critic_loss = nn.MSELoss()(
                        curr_values.squeeze(-1),
                        returns
                     )

                     loss = actor_loss + 0.5 * critic_loss - 0.02 * entropy

                     optimizer.zero_grad()
                     loss.backward()

                     torch.nn.utils.clip_grad_norm_(Network1.parameters(), 5.0)
                     torch.nn.utils.clip_grad_norm_(Network2.parameters(), 5.0)

                     optimizer.step()

                 epoch_rewards.append(rewards.mean().item())

            # =========================
            # Logging
            # =========================
             avg_reward = np.mean(epoch_rewards)
             print(f"Epoch {epoch+1}/{args.epochs} | Reward: {avg_reward:.6f}")

            # =========================
            # Evaluation
            # =========================
             Recall = evaluate(args, Network1, Network2, datasets, test_keys)
             print(f"Recall: {Recall:.6f}")
             if Recall > best_recall:
                 best_recall = Recall

                 torch.save(
                    Network1.state_dict(),
                    osp.join(args.save_path, f'best_model1_subj{args.subject_id}.pth')
                 )
                 torch.save(
                    Network2.state_dict(),
                    osp.join(args.save_path, f'best_model2_subj{args.subject_id}.pth')
                 )

                 print("🔥 Best model saved!")
             Network1.train()
             Network2.train()
             scheduler.step()
    
    else:
    # --- TESTING MODE ---
        datasets = h5py.File(args.deep_features, 'r')
        all_keys = list(datasets.keys())
        test_keys = all_keys[15 * args.subject_id:15 * (args.subject_id + 1)]

        print("# test videos {}.".format(len(test_keys)))
        model1 = ClassifierGNN(in_features=args.n_feature, edge_features=args.edge_features, out_features=args.n_feature, device=DEVICE).to(DEVICE)
        model2 = DSN(in_dim=args.n_feature, hid_dim=args.hid_dim, num_layers=1, cell='gru').to(DEVICE)

        checkpoint_path1 = osp.join(args.save_path, f'best_model1_subj{args.subject_id}.pth')
        checkpoint_path2 = osp.join(args.save_path, f'best_model2_subj{args.subject_id}.pth')
        
        model1.load_state_dict(torch.load(checkpoint_path1, map_location=DEVICE))
        model2.load_state_dict(torch.load(checkpoint_path2, map_location=DEVICE))

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
                seq = torch.from_numpy(datasets[key]['features'][...]).float().to(DEVICE)
                gt = datasets[key]['labels'][...] 
                label_idx = key_idx
                local_label = local_labels[label_idx]

                sub_graphs = []
                for n in range(math.ceil(seq.shape[0] / (args.fragment_length * 2))):
                    chunk = seq[(args.fragment_length * 2) * n : (args.fragment_length * 2) * (n + 1), :]
                    if chunk.shape[0] > 1:
                        sg, _ = model1(chunk)
                        sub_graphs.append(sg)
                    else:
                        sub_graphs.append(chunk)
                
                sub_graphs = torch.cat(sub_graphs, dim=0)
                seq_graph = torch.add(seq, sub_graphs).unsqueeze(0)
                
                sig_probs, _ = model2(seq_graph)
                probs_importance = sig_probs.data.cpu().reshape(-1).numpy()  # reshape(-1) safe for single-frame sequences

                limits = args.num_fragment
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
                                # Extract numeric recall value from the log line
                                recall_val = float(line.split('Recall')[-1].strip())
                                all_recalls.append(recall_val)
                            except ValueError:
                                continue

            if all_recalls:
                mean_recall = np.mean(all_recalls)
                print(f"\n" + "="*30)
                print(f"TESTING SUMMARY (Subject {args.subject_id})")
                print(f"Mean Recall: {mean_recall:.4f}")
                print(f"Log saved to: {log_filepath}")
                print("="*30 + "\n")
            
        # This is the end of the torch.no_grad() block
        datasets.close()