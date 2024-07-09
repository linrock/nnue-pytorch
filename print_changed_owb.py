import hashlib
import os
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import requests

import features
from serialize import NNUEReader, NNUEWriter


feature_set = features.get_feature_set_from_name("HalfKAv2_hm")


def print_spsa_params(nnue_filename1, nnue_filename2):
    with open(nnue_filename1, "rb") as f:
        reader = NNUEReader(f, feature_set)
        base_model = reader.model

    with open(nnue_filename2, "rb") as f:
        reader = NNUEReader(f, feature_set)
        apply_model = reader.model

    c_end_weights = 6
    c_end_biases = 64

    num_weights = 0
    num_biases = 0

    stack_range = range(8)

    # output weights - 8 x 32 = 256
    for i in stack_range:
        for j in range(32):
            if base_model.layer_stacks.output.weight[i, j] != apply_model.layer_stacks.output.weight[i, j]:
                value = round(int(apply_model.layer_stacks.output.weight[i, j] * 600 * 16) / 127)
                print(f"oW[{i}][{j}],{value},-127,127,{c_end_weights},0.0020")
                num_weights += 1

    # output biases - 8
    for j in range(8):
        if base_model.layer_stacks.output.bias[j] != apply_model.layer_stacks.output.bias[j]:
            value = int(apply_model.layer_stacks.output.bias[j] * 600 * 16)
            print(f"oB[{j}],{value},-8192,8192,{c_end_biases},0.0020")
            num_biases += 1

    # print(f"# weights: {num_weights}")
    # print(f"# biases:  {num_biases}")


if __name__ == "__main__":
    print_spsa_params("./nnue/nn-ddcfb9224cdb.nnue", "./nnue/nn-74f1d263ae9a.nnue")
