import cv2
import pickle
import numpy as np
import os

ROOT_PATH = "/kpmlilat/tests/test_embed_lqr_rl/goal_images/cart-pole"
num_frame_stack = 3

imgs_l = []
for i in range(num_frame_stack):
    tmp = cv2.imread(os.path.join(ROOT_PATH, "ezgif-frame-00{}.png".format(i+1)))
    tmp_rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    # Crop the image from 96*96 to size 84*84
    cropped_tmp_rgb = tmp_rgb[6:6+84, 6:6+84]
    imgs_l.append(cropped_tmp_rgb)

goal_obs = np.concatenate(imgs_l, axis=2)

# from (h, w, c) to (c, h, w) for pytorch
goal_obs = np.transpose(goal_obs, (2,0,1))

with open(os.path.join(ROOT_PATH, "goal_obs.pkl"), "wb") as f:
    pickle.dump(goal_obs,f)

