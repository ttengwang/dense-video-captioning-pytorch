# Dense Video Captioning
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dense-captioning-events-in-videos-sysu/dense-video-captioning-on-activitynet)](https://paperswithcode.com/sota/dense-video-captioning-on-activitynet?p=dense-captioning-events-in-videos-sysu)

Code for SYSU submission to [ActivityNet Challenge 2020](http://activity-net.org/challenges/2020/index.html) (Task2: Dense Video Captioning). Our approach follows a two-stage pipeline: first, we extract a set of temporal event proposals;
then we propose a multi-event captioning model to capture the event-level temporal relationships and effectively fuse
the multi-modal information. 

We won the 2nd place and the technical paper is available at [arxiv](https://arxiv.org/abs/2006.11693v2).



# Environment
1. Python 3.6.2
2. CUDA 10.0, [PyTorch 1.2.0](https://pytorch.org/get-started/locally/) (may work on other versions but has not been tested)
3. other modules, run `pip install -r requirement.txt`

# Prerequisites
- ActivityNet video features. We use TSN features following this [repo](https://github.com/salesforce/densecap). You can follow the "Data Preparation" section to download feature files, then decompress and move them into `./data/resnet_bn`.
- Download annotation files and pre-generated proposal files from [Google Drive](https://drive.google.com/drive/folders/1NSL7v7ax-9veJOcLxJpMzFyl5MTCUIUO?usp=sharing), and place them into `./data`. For the proposal generation, please refer to [DBG](https://github.com/Tencent/ActionDetection-DBG) and [ESGN](https://github.com/ttengwang/ESGN).
- Build vocabulary file. Run `python misc/build_vocab.py`. 

- (Optional) You can also test the code based on C3D feature. Download C3D feature files (`sub_activitynet_v1-3.c3d.hdf5`) from [here](http://activity-net.org/challenges/2016/download.html#c3d). Convert the h5 file into npy files and place them into `./data/c3d`.

# Usage

- Training
```
# first, train the model with cross-entropy loss 
cfg_file_path=cfgs/tsrm_cmg_hrnn.yml
python train.py --cfg_path $cfg_file_path

# Afterward, train the model with reinforcement learning on enlarged training set
cfg_file_path=cfgs/tsrm_cmg_hrnn_RL_enlarged_trainset.yml
python train.py --cfg_path $cfg_file_path
```
training logs and generated captions are in this folder `./save`.

- Evaluation
```
# evaluation with ground-truth proposals (small val set with 1000 videos)
result_folder=tsrm_cmg_hrnn_RL_enlarged_trainset
val_caption_file=data/captiondata/expand_trainset/val_1.json
python eval.py --eval_folder $result_folder --eval_caption_file $val_caption_file

# evaluation with learnt proposals (small val set with 1000 videos)
result_folder=tsrm_cmg_hrnn_RL_enlarged_trainset
lnt_tap_json=data/generated_proposals/tsn_dbg_esgn_valset_num4717.json
python eval.py --eval_folder $result_folder --eval_caption_file $val_caption_file --load_tap_json $lnt_tap_json

# evaluation with ground-truth proposals (standard val set with 4917 videos)
result_folder=tsrm_cmg_hrnn
python eval.py --eval_folder $result_folder

# evaluation with learnt proposals (standard val set with 4917 videos)
result_folder=tsrm_cmg_hrnn
lnt_tap_json=data/generated_proposals/tsn_dbg_esgn_valset_num4717.json
python eval.py --eval_folder $result_folder --load_tap_json $lnt_tap_json
```

- Testing
```
python eval.py --eval_folder tsrm_cmg_hrnn_RL_enlarged_trainset \
 --load_tap_json data/generated_proposals/tsn_dbg_esgn_testset_num5044.json\
 --eval_caption_file data/captiondata/fake_test_anno.json
```

We also provide the config files of some **baseline models**. Please see this folder `./cfgs` for details. 


# Pre-trained model

We provide a pre-trained model from [here](https://drive.google.com/drive/folders/1EqQCzjfJSOyKVq_Rzoi0xAcHLJhVlVht?usp=sharing). You can directly download `model-best-RL.pth` and `info.json` and place them into `./save/tsrm_cmg_hrnn_RL_enlarged_trainset`, then run the above code for fast evaluation. On the [small validation set (1000 videos)](data/captiondata/expand_trainset/val_1.json), this model achieves a 14.51/10.14 METEOR with ground-truth/learnt proposals.

# Related project

[PDVC (ICCV 2021)](https://github.com/ttengwang/PDVC): A simple yet effective dense video captioning method, which integrates the proposal generation and captioning generation into a parallel decoding manner. 

# Citation
If you find this repo helpful to your research, please consider citing:
```
@article{wang2020dense,
  title={Dense-Captioning Events in Videos: SYSU Submission to ActivityNet Challenge 2020},
  author={Wang, Teng and Zheng, Huicheng and Yu, Mingjing},
  journal={arXiv preprint arXiv:2006.11693},
  year={2020}
}
```

# References
- Awesome [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) project.
- [Official implementation](https://github.com/XgDuan/WSDEC) of "Weakly Supervised Dense Event Captioning in Videos".
