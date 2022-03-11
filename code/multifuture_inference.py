# coding=utf-8
"""Multifuture inferencing."""

import argparse
import json
import os
import math
import random
#import cPickle as pickle
import pickle
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# the following will still have colocation debug info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from glob import glob
from tqdm import tqdm

from pred_models import Model as PredictionModel
from pred_utils import relative_to_abs

parser = argparse.ArgumentParser()
parser.add_argument("traj_path")
parser.add_argument("multifuture_path")
parser.add_argument("model_path")
parser.add_argument("output_file", help="a pickle, traj_id -> all output")

parser.add_argument("--num_out", default=20, type=int,
                    help="number of output per sample")

parser.add_argument("--save_prob_file", default=None,
                    help="save beam prob to a file")

parser.add_argument("--greedy", action="store_true")
parser.add_argument("--center_only", action="store_true")
parser.add_argument("--cap_reg", action="store_true")

parser.add_argument("--gpuid", type=int, default=0)
parser.add_argument("--obs_length", type=int, default=8)

# ------------------- basic model parameters
parser.add_argument("--emb_size", type=int, default=128)
parser.add_argument("--enc_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--dec_hidden_size", type=int,
                    default=256, help="hidden size for rnn")


parser.add_argument("--grid_strides", default="2,4")
parser.add_argument("--use_grids", default="1,0")

parser.add_argument("--use_gn", action="store_true")
parser.add_argument("--use_gnn", action="store_true")

parser.add_argument("--use_scene_enc", action="store_true")
parser.add_argument("--use_single_decoder", action="store_true")
parser.add_argument("--use_soft_grid_class", action="store_true")

parser.add_argument("--diverse_beam", action="store_true")
parser.add_argument("--diverse_gamma", type=float, default=1.0)
parser.add_argument("--fix_num_timestep", type=int, default=0)

parser.add_argument("--scene_feat_path", default=None)
parser.add_argument("--scene_id2name", default=None)
parser.add_argument("--scene_h", type=int, default=36)
parser.add_argument("--scene_w", type=int, default=64)
parser.add_argument("--scene_class", type=int, default=11)
parser.add_argument("--convlstm_kernel", default=3, type=int)
parser.add_argument("--scene_conv_dim", default=64, type=int)
parser.add_argument("--scene_conv_kernel", default=3, type=int)


# arguments for some new added implementations.
parser.add_argument("--add_self_attn", action="store_true")
parser.add_argument("--add_mr", action="store_true")

# other boxes
parser.add_argument("--add_geo_relations", action="store_true")
parser.add_argument("--random_other", action="store_true",
                    help="randomize top k other boxes")
parser.add_argument("--max_other", type=int, default=20,
                    help="maximum number of other box")
parser.add_argument("--box_emb_size", type=int, default=64)
parser.add_argument("--num_box_class", type=int, default=10)

# goal-driven component.
parser.add_argument("--add_goal", action="store_true")
parser.add_argument("--num_goals", type=int, default=5)
parser.add_argument("--geo_goal", action="store_true")

# exponential loss.
parser.add_argument("--exp_loss", action="store_true")

parser.add_argument("--video_h", type=int, default=1080)
parser.add_argument("--video_w", type=int, default=1920)

path = "./forking_paths_dataset/next_x_v1_dataset_prepared_data/obs_data/"
person_box_path = os.path.join(path, "anno_person_box/test/")
other_box_path = os.path.join(path, "anno_other_box/test/")
person = glob(os.path.join(person_box_path, "*.p"))
videos = glob(os.path.join(other_box_path, "*.p"))

def load_traj(traj_file):
  data = []
  delim = "\t"
  with open(traj_file, "r") as f:
    for line in f:
      fidx, pid, x, y = line.strip().split(delim)
      data.append([fidx, pid, x, y])
  return np.array(data, dtype="float32")


def add_grid(args):
  args.scene_grid_strides = [int(o) for o in args.grid_strides.split(",")]
  assert args.scene_grid_strides
  args.num_scene_grid = len(args.scene_grid_strides)
  args.use_grids = [bool(int(o)) for o in args.use_grids.split(",")]

  args.scene_grids = []
  # the following is consistent with tensorflow conv2d when given odd input
  for stride in args.scene_grid_strides:
    h, w = args.scene_h, args.scene_w
    this_h, this_w = round(h*1.0/stride), round(w*1.0/stride)
    this_h, this_w = int(this_h), int(this_w)
    args.scene_grids.append((this_h, this_w))

  # Get the center point for each scale's each grid
  args.scene_grid_centers = []
  args.grid_box_sizes = []
  for h, w in args.scene_grids:
    h_gap, w_gap = args.video_h*1.0/h, args.video_w*1.0/w
    args.grid_box_sizes.append((h_gap, w_gap))
    centers_x = np.cumsum([w_gap for _ in range(w)]) - w_gap/2.0
    centers_y = np.cumsum([h_gap for _ in range(h)]) - h_gap/2.0
    centers_xx = np.tile(np.expand_dims(centers_x, axis=0), [h, 1])
    centers_yy = np.tile(np.expand_dims(centers_y, axis=1), [1, w])
    centers = np.stack((centers_xx, centers_yy), axis=-1)  # [H,W,2]
    args.scene_grid_centers.append(centers)

def get_grid_input(args, traj):
  # traj is [obs_length, 2]
  grid_class = np.zeros([len(args.scene_grids), args.obs_length], dtype="int32")
  grid_target_all = []
  # get the grid classification label based on (x,y)
  # grid centers: [H,W,2]
  for i, (center, (h, w)) in enumerate(zip(
      args.scene_grid_centers, args.scene_grids)):
    grid_target = np.zeros((args.obs_length, h, w, 2), dtype="float32")
    # grid classification
    h_gap, w_gap = args.video_h*1.0/h, args.video_w*1.0/w
    x_indexes = np.ceil(traj[:, 0] / w_gap)  # [obs_length]

    y_indexes = np.ceil(traj[:, 1] / h_gap)  # [obs_length]
    x_indexes = np.asarray(x_indexes, dtype="int")
    y_indexes = np.asarray(y_indexes, dtype="int")

    # ceil(0.0) = 0.0, we need
    x_indexes[x_indexes == 0] = 1
    y_indexes[y_indexes == 0] = 1
    x_indexes = x_indexes - 1
    y_indexes = y_indexes - 1

    one_hot = np.zeros((args.obs_length, h, w), dtype="uint8")
    one_hot[range(args.obs_length), y_indexes, x_indexes] = 1
    one_hot_flat = one_hot.reshape((args.obs_length, -1))  # [obs_length, h*w]
    classes = np.argmax(one_hot_flat, axis=1)  # [obs_length]
    grid_class[i, :] = classes

    # grid regression
    # tile current person seq xy
    traj_tile = np.tile(np.expand_dims(np.expand_dims(
        traj, axis=1), axis=1), [1, h, w, 1])
    # tile center [obs_length, h, w, 2]
    center_tile = np.tile(np.expand_dims(
        center, axis=0), [args.obs_length, 1, 1, 1])
    # grid_center + target -> actual xy
    all_target = traj_tile - center_tile  # [obs_length, h,w,2]
    # only save the one grid
    grid_target[:, :, :, :] = all_target
    grid_target_all.append(grid_target)
  return grid_class, grid_target_all

def get_inputs(args, traj_files, gt_trajs):
  traj_list = []  # [N] [obs_length, 2]
  traj_list_rel = []  # [N] [obs_length, 2]

  scene_feats = []  # all frame seg
  scene_featidx_list = []  # [N, obs_length, 1]

  grid_class_list = []  # [N, strides, obs_length]
  grid_target_list = []  # [N, strides, obs_length, 2]

  pred_length_list = []  # [N]

  # [N,1] a list of variable number of boxes
  other_box_seq_list = []
  # [N,1] # a list of variable number of boxes classes
  other_box_class_seq_list = []

  # the person traj's boxes
  box_seq_list = []  # [N, seq_len, 4]


  with open(args.scene_id2name, "r") as f:
    scene_id2name = json.load(f)  # {"oldid2new":,"id2name":}
  scene_oldid2new = scene_id2name["oldid2new"]
  scene_oldid2new = {
      int(oldi): scene_oldid2new[oldi] for oldi in scene_oldid2new}
  # for background class or other class that we ignored
  assert 0 not in scene_oldid2new
  scene_oldid2new[0] = 0
  total_scene_class = len(scene_oldid2new)
  scene_id2name = scene_id2name["id2name"]
  scene_id2name[0] = "BG"
  assert len(scene_oldid2new) == len(scene_id2name)
  other_box = []
  other_box_class = []

  # only 1 CA for each scene.
  person_box = np.zeros((507, args.obs_length, 4),
                        dtype="float32")
  # person_boxids = np.zeros((1, args.obs_length),
  #                          dtype="int32")

  print("Preparing traj, person box, other box....")
  count_person = 0
  for traj_file, box_file, person_file in zip(traj_files, videos, person):
    # open traj file, other box file, person box file.
    traj_id = os.path.splitext(os.path.basename(traj_file))[0]
    box_id = os.path.splitext(os.path.basename(box_file))[0]
    person_id = os.path.splitext(os.path.basename(person_file))[0]
    # assert traj_id == box_id
    scene, moment_idx, x_agent_pid, camera = traj_id.split("_")
    scene_box, moment_box, ca_box, cam = box_id.split("_")
    cene_person, moment_person, ca_person, cam_person = person_id.split("_")
    assert moment_idx == moment_box
    assert x_agent_pid == ca_box
    assert moment_idx == moment_person
    assert x_agent_pid == ca_person

    x_agent_pid = int(x_agent_pid)
    ca_box = int(ca_box)
    ca_person = int(ca_person)

    # load all features
    traj_data = load_traj(traj_file)

    # assuming the frameIdx is sorted in ASC
    frame_idxs = np.unique(traj_data[:, 0]).tolist()

    # we only need the x_agent's trajectory
    # [obs_length, 2]
    x_agent_obs_traj = traj_data[x_agent_pid == traj_data[:, 1], 2:]
    assert len(x_agent_obs_traj) == args.obs_length, (
        traj_id, x_agent_obs_traj.shape)

    x_agent_obs_traj_rel = np.zeros_like(x_agent_obs_traj)
    x_agent_obs_traj_rel[1:, :] = x_agent_obs_traj[1:, :] - \
        x_agent_obs_traj[:-1, :]

    # for this trajectory we get all the features
    # 1. grid
    # [scale, obs_length], [2][obs_length, 2]
    grid_class, grid_target = get_grid_input(args, x_agent_obs_traj)

    # 2. person box / other boxes / scene feature
    # print("getting boxes: ")
    with open(box_file, "rb") as f:
      other_boxes = pickle.load(f, encoding="latin1")

    with open(person_file, "rb") as f:
      person_boxes = pickle.load(f, encoding="latin1")

    this_other_box = []
    this_other_box_class = []
    for i, frame_idx in enumerate(frame_idxs):
      key = "%d_%d" % (frame_idx, ca_box)
      # a list of [4]
      # [[b1], [b2]
      #   [b1], [b2]]
      this_other_box.append(other_boxes[key][0])
      #  a list of [1]
      this_other_box_class.append(other_boxes[key][1])
      # print(this_other_box)
    other_box.append(this_other_box)
    other_box_class.append(this_other_box_class)
    # print(len(other_box))

    for i, frame_idx in enumerate(frame_idxs):
      key = "%d_%d" % (frame_idx, ca_person)
      person_box[count_person, i, :] = person_boxes[key]


    scene_featidx = np.zeros([args.obs_length, 1], dtype="int32")
    for i, frame_idx in enumerate(frame_idxs):
      scene_feat_file = os.path.join(args.scene_feat_path, traj_id,
                                     "%s_F_%08d.npy" % (traj_id, frame_idx))
      feati = len(scene_feats)
      # get the feature new i
      scene_feats.append(np.load(scene_feat_file))
      scene_featidx[i, :] = feati


    
    # pack up all the features
    traj_list.append(x_agent_obs_traj)
    traj_list_rel.append(x_agent_obs_traj_rel)

    scene_featidx_list.append(scene_featidx)

    grid_class_list.append(grid_class)
    grid_target_list.append(grid_target)

    # get the multifuture maximum pred timestep
    pred_timesteps = [len(gt_trajs[traj_id][future_id]["x_agent_traj"])
                      for future_id in gt_trajs[traj_id]]
    pred_length_list.append(max(pred_timesteps))

    count_person += 1

  box_seq_list.append(person_box[:count_person])
  box_seq_list = np.concatenate(box_seq_list, axis=0)
  print("person box length: " + str(len(box_seq_list)))
  assert(len(box_seq_list)) == 507

  # print(len(other_box))
  other_box_seq_list.extend(other_box)
  other_box_class_seq_list.extend(other_box_class)
  print("other box length: " + str(len(other_box_seq_list)))
  assert len(other_box_seq_list) == 507

  other_box_seq_list = np.asarray(
    other_box_seq_list)  # [N*K,seqlen] list type?
  other_box_class_seq_list = np.asarray(
    other_box_class_seq_list)  # [N*K,seqlen] list type?

  # replace the scene feature
  scene_feat_shape = [len(scene_feats), args.scene_h, args.scene_w,
                      total_scene_class]
  scene_feat_all = np.zeros(scene_feat_shape, dtype="uint8")
  print("making scene feature of shape %s..." % (scene_feat_shape))
  for k, scene_feat in tqdm(enumerate(scene_feats), total=len(scene_feats)):
    # transform classid first
    new_scene_feat = np.zeros_like(scene_feat)  # zero for background class
    for i in range(args.scene_h):
      for j in range(args.scene_w):
        # rest is ignored and all put into background
        if scene_feat[i, j] in scene_oldid2new:
          new_scene_feat[i, j] = scene_oldid2new[scene_feat[i, j]]
    # transform to masks
    this_scene_feat = np.zeros(
        (args.scene_h, args.scene_w, total_scene_class), dtype="uint8")
    # so we use the H,W to index the mask feat
    # generate the index first
    h_indexes = np.repeat(np.arange(args.scene_h), args.scene_w).reshape(
        (args.scene_h, args.scene_w))
    w_indexes = np.tile(np.arange(args.scene_w), args.scene_h).reshape(
        (args.scene_h, args.scene_w))
    this_scene_feat[h_indexes, w_indexes, new_scene_feat] = 1

    scene_feat_all[k, :, :, :] = this_scene_feat
    del this_scene_feat
    del new_scene_feat
  print("Done.")
  return {
      "obs_traj": traj_list,
      "obs_traj_rel": traj_list_rel,

      "obs_grid_class": grid_class_list,
      "obs_grid_target": grid_target_list,

      "obs_scene": scene_featidx_list,

      "scene_feats": scene_feat_all,
      "max_pred_lengths": pred_length_list,

      "obs_other_box": other_box_seq_list,
      "obs_other_box_class": other_box_class_seq_list,

      "obs_box": box_seq_list
  }


def encode_other_boxes(self, person_box, other_box):
  """Encoder other boxes."""
  # get relative geometric feature
  x1, y1, x2, y2 = person_box
  xx1, yy1, xx2, yy2 = other_box

  x_m = x1
  y_m = y1
  w_m = x2 - x1
  h_m = y2 - y1

  x_n = xx1
  y_n = yy1
  w_n = xx2 - xx1
  h_n = yy2 - yy1

  return [
      math.log(max((x_m - x_n), 1e-3)/w_m),
      math.log(max((y_m - y_n), 1e-3)/h_m),
      math.log(w_n/w_m),
      math.log(h_n/h_m),
  ]

def load_model_weights(model_path, sess, top_scope=None):
  """Load model weights into tf Graph."""

  tf.global_variables_initializer().run()
  allvars = tf.global_variables()
  allvars = [var for var in allvars if "global_step" not in var.name]
  restore_vars = allvars
  opts = ["Adam", "beta1_power", "beta2_power", "Adam_1", "Adadelta_1",
          "Adadelta", "Momentum"]
  restore_vars = [var for var in restore_vars
                  if var.name.split(":")[0].split("/")[-1] not in opts]

  if top_scope is not None:
    restore_vars = [var for var in restore_vars
                    if var.name.split(":")[0].split("/")[0] == top_scope]
  saver = tf.train.Saver(restore_vars, max_to_keep=5)

  load_from = model_path

  ckpt = tf.train.get_checkpoint_state(load_from)
  if ckpt and ckpt.model_checkpoint_path:
    loadpath = ckpt.model_checkpoint_path
    saver.restore(sess, loadpath)
  else:
    raise Exception("Model not exists")

class PredictionModelInference(PredictionModel):
  """Rewrite the future prediction model for inferencing."""

  def get_feed_dict(self, inputs, args, idx):
    """Givng a batch of data, construct the feed dict."""

    # Tensor dimensions, so pylint: disable=g-bad-name
    N = 1

    T_in = args.obs_length
    T_pred = inputs["max_pred_lengths"][idx]

    feed_dict = {}
    obs_length = np.zeros((N), dtype="int32")
    pred_length = np.zeros((N), dtype="int32")
    feed_dict[self.obs_length] = obs_length
    feed_dict[self.pred_length] = pred_length
    obs_length[0] = T_in
    pred_length[0] = T_pred

    feed_dict[self.is_train] = False

    for j, (h, w) in enumerate(args.scene_grids):
      if not args.use_grids[j]:
        continue
      grid_obs_labels = np.zeros([1, T_in], dtype="int")
      grid_obs_reg_targets = np.zeros([1, T_in, h, w, 2], dtype="float")


      grid_obs_labels[0, :] = inputs["obs_grid_class"][idx][j, :]

      grid_obs_reg_targets[0, :, :, :, :] = \
          inputs["obs_grid_target"][idx][j][:, :, :, :]

      feed_dict[self.grid_obs_labels[j]] = grid_obs_labels
      feed_dict[self.grid_obs_regress[j]] = grid_obs_reg_targets

      feed_dict[self.grid_pred_regress[j]] = np.zeros(
          [1, T_pred, h, w, 2], dtype="float")
      if args.use_soft_grid_class:
        feed_dict[self.grid_pred_labels_T[j]] = np.zeros(
            [1, T_pred, h, w, 1], dtype="int")
      else:
        feed_dict[self.grid_pred_labels_T[j]] = np.zeros(
            [1, T_pred], dtype="int")

    # reconstruct the scene feature first
    oldid2newid = {}
    new_scene_idxs = np.zeros([1, T_in, 1], dtype="int32")
    for j in range(T_in):
      oldid = inputs["obs_scene"][idx][j][0]
      if oldid not in oldid2newid:
        oldid2newid[oldid] = len(oldid2newid)
      newid = oldid2newid[oldid]
      new_scene_idxs[0, j, 0] = newid
    # get all the feature used by this mini-batch
    scene_feat = np.zeros((len(oldid2newid), args.scene_h,
                           args.scene_w, args.scene_class),
                          dtype="float32")
    for oldid in oldid2newid:
      newid = oldid2newid[oldid]
      scene_feat[newid, :, :, :] = \
          inputs["scene_feats"][oldid, :, :, :]

    # initial all the placeholder
    obs_scene = np.zeros((N, T_in), dtype="int32")
    obs_scene_mask = np.zeros((N, T_in), dtype="bool")
    feed_dict[self.obs_scene] = obs_scene
    feed_dict[self.obs_scene_mask] = obs_scene_mask
    feed_dict[self.scene_feat] = scene_feat

    # each bacth
    for j in range(T_in):
      # it was (1) shaped
      obs_scene[0, j] = new_scene_idxs[0, j, 0]
      obs_scene_mask[0, j] = True

    # [N,num_scale, T] # each is int to num_grid_class
    for j, _ in enumerate(args.scene_grids):
      this_grid_label = np.zeros([N, T_in], dtype="int32")
      this_grid_label[0, :] = inputs["obs_grid_class"][idx][j, :]

      feed_dict[self.grid_obs_labels[j]] = this_grid_label


    if args.add_geo_relations:
      # todo: now is geo_relation feed in.
      KB = args.max_other
      num_box_class = args.num_box_class
      other_boxes_class = np.zeros(
        (N, T_in, KB, num_box_class), dtype='float32')
      other_boxes = np.zeros((N, T_in, KB, 4), dtype='float32')
      other_boxes_mask = np.zeros((N, T_in, KB), dtype='bool')

      # for i in range(len(inputs['obs_other_box'])):
      for j in range(len(inputs['obs_other_box'][idx])):  # -> seq_len
        this_other_boxes = inputs['obs_other_box'][idx][j]
        this_other_boxes_class = inputs['obs_other_box_class'][idx][j]

        other_box_idxs = range(len(this_other_boxes))

        if args.random_other:
          random.shuffle(other_box_idxs)

        other_box_idxs = other_box_idxs[:KB]

        # get the current person box
        this_person_x1y1x2y2 = inputs['obs_box'][idx][j]  # (4)

        for k, this_idx in enumerate(other_box_idxs):
          other_boxes_mask[0, j, k] = True

          other_box_x1y1x2y2 = this_other_boxes[this_idx]

          other_boxes[0, j, k, :] = self.encode_other_boxes(
            this_person_x1y1x2y2, other_box_x1y1x2y2)
          # one-hot representation
          box_class = this_other_boxes_class[this_idx]
          other_boxes_class[0, j, k, box_class] = 1
      # print(other_boxes)
      feed_dict[self.obs_other_boxes] = other_boxes
      feed_dict[self.obs_other_boxes_class] = other_boxes_class
      feed_dict[self.obs_other_boxes_mask] = other_boxes_mask

    return feed_dict

if __name__ == "__main__":

  args = parser.parse_args()

  add_grid(args)
  args.use_beam_search = True
  if args.greedy:
    args.use_beam_search = False
  assert sum(args.use_grids) == 1


  # get all the test data
  traj_files = glob(os.path.join(args.traj_path, "*.txt"))
  traj_ids = [os.path.splitext(os.path.basename(one))[0] for one in traj_files]
  gt_trajs = {}
  for traj_id in traj_ids:
    with open(os.path.join(args.multifuture_path, "%s.p" % traj_id), "rb") as f:
      gt_trajs[traj_id] = pickle.load(f)

  # get all the preprocessed data input
  inputs = get_inputs(args, traj_files, gt_trajs)

  output_data = {}  # traj_id ->
  beam_prob = {} # traj_id ->

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (",".join([
      "%s" % i for i in [args.gpuid]]))

  with tf.Session(config=tfconfig) as sess:
    # load model
    model_config = argparse.Namespace(
        modelname="model",
        batch_size=1,

        beam_size=args.num_out,
        use_beam_search=args.use_beam_search,
        diverse_beam=args.diverse_beam,
        diverse_gamma=args.diverse_gamma,
        fix_num_timestep=args.fix_num_timestep,

        use_teacher_forcing=False,
        is_train=False,

        scene_h=args.scene_h,
        scene_w=args.scene_w,
        scene_class=args.scene_class,
        use_soft_grid_class=args.use_soft_grid_class,
        use_single_decoder=args.use_single_decoder,

        pred_len=12,
        emb_size=args.emb_size,
        enc_hidden_size=args.enc_hidden_size,
        dec_hidden_size=args.dec_hidden_size,
        activation_func=tf.nn.tanh,
        scene_conv_kernel=args.scene_conv_kernel,
        use_scene_enc=args.use_scene_enc,
        scene_conv_dim=args.scene_conv_dim,
        convlstm_kernel=args.convlstm_kernel,
        use_gnn=args.use_gnn,

        add_self_attn=args.add_self_attn,
        add_mr=args.add_mr,
        add_geo_relations=args.add_geo_relations,
        random_other=args.random_other,
        max_other=args.max_other,
        box_emb_size=args.box_emb_size,
        num_box_class=args.num_box_class,

        add_goal=args.add_goal,
        num_goals=args.num_goals,
        geo_goal=args.geo_goal,

        exp_loss=args.exp_loss,


        keep_prob=1.0,
        scene_grid_strides=args.scene_grid_strides,
        scene_grids=args.scene_grids,
        use_grids=args.use_grids)

    with tf.device("/gpu:%s" % args.gpuid):
      model = PredictionModelInference(model_config, model_config.modelname)
    # print(tf.train.list_variables(model))
    load_model_weights(args.model_path, sess, top_scope="person_pred")

    use_grid_idx = args.use_grids.index(True)
    h_size, w_size = args.grid_box_sizes[use_grid_idx]
    for i, traj_id in tqdm(enumerate(traj_ids), total=len(traj_ids)):
      feed_dict = model.get_feed_dict(inputs, args, i)

      # prediction output
      output_tensors = [model.grid_pred_decoded[use_grid_idx],
                        model.grid_pred_reg_decoded[use_grid_idx]]
      if args.use_beam_search:
        output_tensors += [model.beam_outputs]
      else:
        output_tensors += [model.grid_pred_decoded[use_grid_idx]]

      class_output, reg_output, beam_outputs = \
          sess.run(output_tensors, feed_dict=feed_dict)


      out_trajs = []
      pred_len = inputs["max_pred_lengths"][i]
      # [N, T, H*W, 2]
      reg_output = reg_output.reshape([1, pred_len, -1, 2])

      center_coors = args.scene_grid_centers[use_grid_idx] # [H, W, 2]
      center_coors = center_coors.reshape([-1, 2])  # [H*W, 2]

      if args.greedy:
        # get greedy decoder output
        class_output = class_output.reshape([1, pred_len, -1])
        # [1, T]
        class_output_selected = np.argmax(class_output, axis=2)
        greedy_out_traj = []
        for t in range(pred_len):
          this_grid_class = class_output_selected[0, t]
          this_center = center_coors[this_grid_class]
          this_reg = reg_output[0, t, this_grid_class, :]
          if args.center_only:
            pred_point = this_center
          else:
            pred_point = this_center + this_reg
          greedy_out_traj.append(pred_point)
        out_trajs = [greedy_out_traj for _ in range(args.num_out)]
      else:
        beam_logits, beam_grid_ids, beam_logprobs = beam_outputs
        output_logits = beam_logits
        #print(reg_output.shape)  # [1, T, h, w, 2]
        #print(beam_grid_ids.shape)  # [1, beam_size, T]
        for j in range(args.num_out):
          this_traj = []
          for t in range(pred_len):
            this_grid_class = beam_grid_ids[0, j, t]
            this_center = center_coors[this_grid_class]
            this_reg = reg_output[0, t, this_grid_class, :]
            if args.center_only:
              pred_point = this_center
            else:

              pred_point = this_center + this_reg

            this_traj.append(pred_point)
          out_trajs.append(this_traj)

      # save the output
      output_data[traj_id] = out_trajs
      if args.save_prob_file is not None:
        # [1, beam_size, T, H*W] and beam_size of prob
        beam_prob[traj_id] = (beam_logits, beam_logprobs)

  with open(args.output_file, "wb") as f:
    pickle.dump(output_data, f)

  if args.save_prob_file is not None:
    with open(args.save_prob_file, "wb") as f:
      pickle.dump(beam_prob, f)
