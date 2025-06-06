import os
import torch
from rsl_rl.modules import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87

path = '/home/zhanghb2023/project/extreme-parkour/legged_gym/logs/parkour_new/329-12-38/traced'
base_model_name = '329-12-38-10000-base_jit.pt'
vision_model_name = '329-12-38-10000-vision_weight.pt'

base_model_path = os.path.join(path, base_model_name)
vision_model_path = os.path.join(path, vision_model_name)

base_model = torch.jit.load(base_model_path)

vision_model = torch.load(vision_model_path)

base_model.eval()
# vision_model.eval()

# priv_encoder, history_encoder, scan_encoder, actor, estimator = base_model
# print('base model: ', base_model)

# priv_encoder: 29 * 20 这个是为了训练 estimator，实际上不使用
# scan_encoder: 132 * 32，# scandot -> scandot latent
# scan_encoder 也是已经用 depth_encoder 训练好了，实际也不需要

# estimator: 53 * 9  # proprio -> priv_explicit
# history_encoder: 53 * 20，因为做了 reshape 操作 # history proprio -> priv_latent
# depth_encoder: 58,87 * 32
# actor: 114 (53 + 20 + 9 + 32) * 12

# print('base model: ')
# for name, param in base_model.named_parameters():
#     print(name, param.size())

# priv_encoder = base_model.estimator.estimator
# priv_encoder.eval()
# proprio = torch.ones(1, 53)
# priv_explicit = priv_encoder(proprio)
# print('proprio: ', priv_explicit)

# print('vision model type: ', type(vision_model))
# print('vision model keys: ', vision_model.keys())

depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
depth_encoder = RecurrentDepthBackbone(depth_backbone, None)
depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])

print('vision model: ')
for name, param in depth_encoder.named_parameters():
    print(name, param.shape)
