# to-kpm
The official code for the paper "Task-Oriented Koopman-Based Control with Contrastive Encoder"

## Preparation
1. Install conda and create a conda environment (python 3.7.13) \
     `conda create env --name kpmlilat `
2. Install necessary dependencies \
     `pip install -r requirements.txt`
3. Verify if the simulator dmc2gym works, referring to https://github.com/denisyarats/dmc2gym.

## Supported RL tasks 
Simulated environments of DeepMind dm_control.
- CartPole Swingup 4D
- Cheetah Running 18D
- CartPole Swingup Pixel 

## Usage
1. For training, check `test_one_staged_lqr_pixel.py` and run `run.sh`. Remember to change the CUDA available devices to your setup, and modify the paths included in the config file (under the folder `config`).
2. For evalution, check `test_one_staged_lqr_pixel_only_evaluate.py`.
3. All utilities functions are in `utils.py`. All plot functions are in `paperplot.py` and `rebuttalplot.py`
4. Use autoencoder other than contrastive encoder, run `test_one_staged_lqr_pixel_AE` (performance not good, mainly as a comparison).
5. Two-stage approach test is in `test_two_staged_lqr.py` (not fully tested, performance not good, mainly as a comparison)

## Useful references and codebases
[1] Laskin, Michael, Aravind Srinivas, and Pieter Abbeel. "Curl: Contrastive unsupervised representations for reinforcement learning." International Conference on Machine Learning. PMLR, 2020. (https://github.com/MishaLaskin/curl)

[2] Yin, H., Welle, M. C., and Kragic, D. Embedding Koopman Optimal Control in Robot Policy Learning. 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). (https://github.com/navigator8972/koopman_policy)

## Cite the paper
```
@article{lyu2023task,
  title={Task-Oriented Koopman-Based Control with Contrastive Encoder},
  author={Lyu, Xubo and Hu, Hanyang and Siriya, Seth and Pu, Ye and Chen, Mo},
  journal={arXiv preprint arXiv:2309.16077},
  year={2023}
}
```

## Contact the author
Please feel free to leave any question on github issues or contact me via email lvxubo92 at gmail dot com.