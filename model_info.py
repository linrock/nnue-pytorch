import torch
from torch import nn

import features
from serialize import NNUEReader


feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
with open("nn-ddcfb9224cdb.nnue", "rb") as f:
    reader = NNUEReader(f, feature_set)
    model = reader.model


# -- input --
model.input.weight.shape == torch.Size([22528, 3080])
# 3080 = 3072 + 8 psqt

model.input.bias.shape == torch.Size([3080])
# ftB[3080] = {-120, 102, -112, ... , 26, -148, -18};
model.input.bias * 254  # 127 * 2 ?


# -- L1 --
# model.layer_stacks.l1
nn.Linear(in_features=3072, out_features=128, bias=True)
# out_features = 128 = (8 layer stacks) * (16 outputs per stack)

model.layer_stacks.l1.weight.shape == torch.Size([128, 3072])
# onew[393216] = {5, -4, 4, ... , -6, 14, -9};
model.layer_stacks.l1.weight * 64

model.layer_stacks.l1.bias.shape == torch.Size([128])
# oneb[128] = {-2684, 7895, -6, 708, ... , -1327, -2337, -1242, 162};
model.layer_stacks.l1.bias * 64 * 127  # 8128


# -- L2 --
# model.layer_stacks.l2
nn.Linear(in_features=30, out_features=256, bias=True)

# 8 * 32
model.layer_stacks.l2.weight.shape == torch.Size([256, 30])
# twow[7680] = {8, 2, -6, -5, 1, ... , -16, -50, 4, 10};
model.layer_stacks.l2.weight * 64

model.layer_stacks.l2.bias.shape == torch.Size([256])
# twob[256] = {3803, 2364, 1945, -4624, ... , 995, -3915, 1789, -1853};
model.layer_stacks.l2.bias * 64 * 127  # 8128


# -- L3 / output --
# model.layer_stacks.output
nn.Linear(in_features=32, out_features=8, bias=True)

model.layer_stacks.output.weight.shape == torch.Size([8, 32])
# ow[256] = {74, -60, -70, -125, ... , 28, 34, 9, -27};
model.layer_stacks.output.weight * 75.5

model.layer_stacks.output.bias.shape == torch.Size([8])
# ob[8] = {3005, 1253, 2164, 20, 1025, 1711, 1573, 338};
model.layer_stacks.output.bias * 600 * 16  # 9600
