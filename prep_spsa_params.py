import hashlib
import os
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import requests

import features
from serialize import NNUEReader, NNUEWriter


def prep_ft_biases(model):
    c_end = 16
    for i,value in enumerate(model.input.bias.data[:3072]):
        value_int = int(value * 254)
        if value_int < -250 or value_int > 250:
            print(f"ftB[{i}],{value_int},-1024,1024,{c_end},0.0020")


def prep_l2_weights(model):
    c_end = 6
    num_weights = 0
    for i in range(8):
        for j in range(32):
            for k in range(30):
                value = int(model.layer_stacks.l2.weight[32 * i + j, k] * 64)
                if abs(value) > 60 and abs(value) < 124:
                    print(f"twoW[{i}][{j}][{k}],{value},-127,127,{c_end},0.0020")
                    num_weights += 1
    #     print()
    # print(f"# weights to tune: {num_weights}")


def prep_l2_weights_stack0(model):
    c_end = 6
    num_weights = 0
    i = 0
    for j in range(32):
        for k in range(30):
            value = int(model.layer_stacks.l2.weight[32 * i + j, k] * 64)
            print(f"twoW[{i}][{j}][{k}],{value},-127,127,{c_end},0.0020")
            num_weights += 1
    # for j in range(32):
    #     print(f"twoB[{i}][{j}][{k}],{value},-127,127,{c_end},0.0020")
    #     num_weights += 1

    #     print()
    # print(f"# weights to tune: {num_weights}")


def prep_spsa_params(nnue_filename):
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        reader = NNUEReader(f, feature_set)
        model = reader.model
    prep_l2_weights_stack0(model)


if __name__ == "__main__":
    prep_spsa_params("nn-ddcfb9224cdb.nnue")
