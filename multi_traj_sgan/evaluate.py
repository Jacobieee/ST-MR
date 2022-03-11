from glob import glob
import numpy as np

import argparse
import os
import torch

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name", default="forking_paths", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=25, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--batch_size", default=1, type=int)

parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=25,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade, fde


def evaluate(args, loader, generator):
    all_traj = np.zeros([478,20,25,2])

    total_traj = 0
    with torch.no_grad():
        p = 0
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                # loss_mask,
                pid,
                seq_start_end,

            ) = batch

            
            
            p_id = pid.cpu().numpy()
            np.save("pid.npy", p_id)
            
            all_samples = np.zeros([20,25,2])
            n_sample = 0
            for _ in range(args.num_samples):
                # print(seq_start_end)
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                pred_traj_fake = pred_traj_fake.cpu().numpy()
                temp = np.transpose(pred_traj_fake, (1,0,2))
                traj_temp = temp[int(p_id[p])]
                all_samples[n_sample] = traj_temp

                n_sample += 1
            all_traj[p] = all_samples

            p += 1

        assert all_traj.shape == (478, 20, 25, 2)
        np.save('sgan_out.npy', all_traj)


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    
    for path in paths:
      checkpoint = torch.load(path)
      generator = get_generator(checkpoint)
      path = get_dset_path(args.dataset_name, args.dset_type)
      print("Dataset loaded: "+ path)
      _, loader = data_loader(args, path)
      evaluate(args, loader, generator)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
