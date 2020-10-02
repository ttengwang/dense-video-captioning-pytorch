# Event Sequence Generation Network
Uoffical Re-implementation of Event Sequence Selection Network (ESGN) in paper titled "[streamlined dense video captioning](https://arxiv.org/abs/1904.03870v1)". Note that we do not adopt SST to encode frame-level features, which is different from the original model.

# Environment
1. Python 3.6.2
2. CUDA 10.0, [PyTorch 1.2.0](https://pytorch.org/get-started/locally/) (may work on other versions but has not been tested)
3. other modules, run `pip install -r requirement.txt`

# Prerequisites
- C3D feature. Download C3D feature files (`sub_activitynet_v1-3.c3d.hdf5`) from [here](http://activity-net.org/challenges/2016/download.html#c3d). Convert the h5 file into npy files and place them into `./data/c3d`.

# Usage

- Training 
```
cfg_path=cfgs/esgn.yml
python train_RL.py --path_opt $cfg_path
```
- Validation
```
python eval.py --eval_folder esgn_run0 
```
validation with re-ranking,
```
python eval.py --eval_folder esgn_run0 --eval_esgn_rerank 1 
```


# References
- Awesome [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) project.
- [Official implementation](https://github.com/XgDuan/WSDEC) of "Weakly Supervised Dense Event Captioning in Videos".
