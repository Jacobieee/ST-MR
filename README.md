# ST-MR
[Graph-based Spatial Transformer with Memory Replay for Multi-future Pedestrian Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Graph-Based_Spatial_Transformer_With_Memory_Replay_for_Multi-Future_Pedestrian_Trajectory_CVPR_2022_paper.html).

This paper has been accepted by CVPR 2022.

If you find this work useful for your research, please cite:
'''
@InProceedings{Li_2022_CVPR,
    author    = {Li, Lihuan and Pagnucco, Maurice and Song, Yang},
    title     = {Graph-Based Spatial Transformer With Memory Replay for Multi-Future Pedestrian Trajectory Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {2231-2241}
}
'''

## Dataset
The dataset we use is [VIRAT/ActEV](https://actev.nist.gov/) and [The Forking Paths](https://github.com/JunweiLiang/Multiverse).

## Preprocessing.
### VIRAT/ActEV.
Follow the [preprocessing step](https://github.com/JunweiLiang/Multiverse/blob/master/TRAINING.md#step-1-preprocess) in Multiverse repo and generate three .npz files with all trajectory inputs and features. Note that this step can occupy large memory. It's recommended to run it on a machine with at least 16g RAM. Or you can prepare the dataset separately.
### The Forking Paths.
Follow [here](https://github.com/JunweiLiang/Multiverse/blob/master/forking_paths_dataset/README.md#annotations) to download and prepare the dataset and visualize it. 

## Training

Requirements are listed in requirements.txt.

```
python code/train.py actev_preprocess models my_model/ \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 \
--enc_hidden_size 256 --dec_hidden_size 256 --activation_func tanh \
--keep_prob 1.0 --num_epochs 80 --batch_size 20 --init_lr 0.3 --use_gnn \
--use_scene --learning_rate_decay 0.95 --num_epoch_per_decay 2.0 \
--grid_loss_weight 1.0 --grid_reg_loss_weight 0.2 --save_period 2000 \
--scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 \
--scene_grid_strides 2,4 --use_grids 1,1 --val_grid_num 0 --train_w_onehot --gpuid 0 \
--add_self_attn --add_mr
```

## Testing single-future prediction
```
python code/test.py actev_preprocess models my_model/ \
--wd 0.001 --runId 0 --obs_len 8 --pred_len 12 --emb_size 32 \
--enc_hidden_size 256 --dec_hidden_size 256 --activation_func tanh \
--keep_prob 1.0 --num_epochs 80 --batch_size 20 --init_lr 0.3 --use_gnn \
--use_scene --learning_rate_decay 0.95 --num_epoch_per_decay 2.0 \
--grid_loss_weight 1.0 --grid_reg_loss_weight 0.2 --save_period 2000 \
--scene_h 36 --scene_w 64 --scene_conv_kernel 3 --scene_conv_dim 64 \
--scene_grid_strides 2,4 --use_grids 1,0 --val_grid_num 0 --gpuid 0 \
--add_self_attn --add_mr --load_best
```

## Testing multi-future prediction
### Run inference and visualization, please refer to [here](https://github.com/JunweiLiang/Multiverse/blob/master/TESTING.md#multi-future-trajectory-prediction).


### run minADEk and minFDEk, as well as generate a trajectory usage file named "traj_usage". This is a pickles file that contains the PTU result.
```
python code/multifuture_eval_trajs.py forking_paths_dataset/next_x_v1_dataset_prepared_data/multifuture/test/ \
my_traj.traj.p traj_usage
```

### Run Negative Log-likelihood.
```
python code/multifuture_eval_trajs_prob.py forking_paths_dataset/next_x_v1_dataset_prepared_data/multifuture/test/ \
my_prob.prob.p
```

### For visualization, please refer to [here](https://github.com/JunweiLiang/Multiverse/blob/master/TESTING.md#visualization).
Images in folder "imgs" show a set of comparison among our model, Multiverse, Social GAN and STGAT.


## Run multi-future inference on [sgan](https://github.com/agrimgupta92/sgan) and [stgat](https://github.com/huang-xx/STGAT).
Train the models of sgan and stgat and replace the files in these two repos with those in multi_traj_sgan and multi_traj_stgat. 
For example, if run such inference on stgat, 
1. unzip the forking_paths.zip under the corresponding folder and put it in the "dataset" folder in stgat repo. This file contains the observed trajectories.
2. Run evaluate.py:```python evaluate.py```. It will create a file stgat_out.npy which contains the output trajectories and files.npy which contains the name of all data sample in the Forking Paths dataset.
3. Run ```python get_final_output.py``` to process the output with the same format as my_traj.traj.p. It will create a file stgat_output.traj.p.
4. Run ```python code/multifuture_eval_trajs.py forking_paths_dataset/next_x_v1_dataset_prepared_data/multifuture/test/ \
stgat_output_vp.traj.p``` to get the result. The later steps are the same as the above sections.


## Acknowledgements
Code in this repo is largely borrowed from [Multiverse](https://github.com/JunweiLiang/Multiverse).

