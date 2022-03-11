# coding=utf-8
"""Given the multifuture trajectory output, compute ADE/FDE/PTU"""

import argparse
import os
import pickle
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("gt_path")
parser.add_argument("prediction_file")
parser.add_argument("traj_usage_file")


def get_min(errors):
  # num_pred, [pred_len] / [1]
  sums = [sum(e) for e in errors]
  min_sum = min(sums)
  min_idx = sums.index(min_sum)
  return errors[min_idx], min_idx

if __name__ == "__main__":
  args = parser.parse_args()

  with open(args.prediction_file, "rb") as f:
    prediction = pickle.load(f)

  # error per ground truth trajectory
  ade_errors = {
      "45-degree": [],
      "top-down": [],
      "all": []
  }
  fde_errors = {
      "45-degree": [],
      "top-down": [],
      "all": []
  }

  crowd_ade = {
    "45-degree": [],
    "top-down": [],
    "all": []
  }

  crowd_fde = {
    "45-degree": [],
    "top-down": [],
    "all": []
  }


  """
  get something.
  """
  # PTU.
  usage_list = []
  ade_usage = np.zeros(507)
  fde_usage = np.zeros(507)
  ind = 0
  for traj_id in tqdm(prediction):
    # calculate usage percentage for every scene.
    num_gt, num_chosen_ade, num_chosen_fde = 0, 0, 0
    ade_chosen_list = []
    fde_chosen_list = []

    scene = traj_id.split("_")[0]
    camera = traj_id.split("_")[-1]
    gt_file = os.path.join(args.gt_path, "%s.p" % traj_id)
    with open(gt_file, "rb") as f:
      gt = pickle.load(f)

    # for each ground truth possibilities, get the minimum ADE/FDE prediction's
    # as this error
    for future_id in gt:
      gt_traj = gt[future_id]["x_agent_traj"]  # (frameIdx, pid, x, y)
      gt_traj = np.array([one[2:] for one in gt_traj])
      pred_len = len(gt_traj)

      # compute ADE and FDE for all prediction
      this_ade_errors = []  # [num_pred] [pred_len]
      this_fde_errors = []  # [num_pred] [1]
      for pred_out in prediction[traj_id]:
        # print(len(pred_out), pred_len)
        assert len(pred_out) >= pred_len
        diff = gt_traj - pred_out[:pred_len]  # [pred_len, 2]
        diff = diff**2
        diff = np.sqrt(np.sum(diff, axis=1))  # [pred_len]
        this_ade_errors.append(diff.tolist())
        this_fde_errors.append([diff[-1]])

      num_gt += 1


      # [pred_len]
      min_ade_errors, min_ade_traj_idx = get_min(this_ade_errors)

      if min_ade_traj_idx not in ade_chosen_list:
        num_chosen_ade += 1
        ade_chosen_list.append(min_ade_traj_idx)

      # [1]
      min_fde_errors, min_fde_traj_idx = get_min(this_fde_errors)

      if min_fde_traj_idx not in fde_chosen_list:
        num_chosen_fde += 1
        fde_chosen_list.append(min_fde_traj_idx)

      if camera == "cam4":
          ade_errors["top-down"] += min_ade_errors
          fde_errors["top-down"] += min_fde_errors
          if scene == "eth" or scene == "hotel" or scene == "zara01":
            crowd_ade["top-down"] += min_ade_errors
            crowd_fde["top-down"] += min_fde_errors

      else:
          ade_errors["45-degree"] += min_ade_errors
          fde_errors["45-degree"] += min_fde_errors
          if scene == "eth" or scene == "hotel" or scene == "zara01":
            crowd_ade["45-degree"] += min_ade_errors
            crowd_fde["45-degree"] += min_fde_errors

      ade_errors["all"] += min_ade_errors
      fde_errors["all"] += min_fde_errors
      if scene == "eth" or scene == "hotel" or scene == "zara01":
        crowd_ade["all"] += min_ade_errors
        crowd_fde["all"] += min_fde_errors

    # calculate usage.
    usage_list.append((num_chosen_ade/num_gt, num_chosen_fde/num_gt))
    ade_usage[ind] = num_chosen_ade/num_gt
    fde_usage[ind] = num_chosen_fde/num_gt
    ind += 1

  # with open("traj_usage", "wb") as f:
  #   pickle.dump(usage_list, f)

  avg_ade_use = np.mean(ade_usage)
  avg_fde_use = np.mean(fde_usage)
  save_use = {
    "ade_use": avg_ade_use,
    "fde_use": avg_fde_use
  }
  with open(args.traj_usage_file, "wb") as f:
    pickle.dump(save_use, f)

  print("ADE/FDE in all scenes (first row) /crowded scenes (second row):")
  keys = ["45-degree", "top-down", "all"]
  print(" ".join(keys + keys))
  print(" ".join(["%s" % np.mean(ade_errors[k])
                  for k in keys] + ["%s" % np.mean(fde_errors[k])
                  for k in keys]))
  print(" ".join(["%s" % np.mean(crowd_ade[k])
                  for k in keys] + ["%s" % np.mean(crowd_fde[k])
                  for k in keys]))
            
  
