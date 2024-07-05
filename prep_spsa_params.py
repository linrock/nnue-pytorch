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
        if abs(value_int) > 200:
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


def prep_l2_weights(model):
    c_end = 6
    num_weights = 0
    for i in range(8):
        for j in range(32):
            for k in range(30):
                value = int(model.layer_stacks.l2.weight[32 * i + j, k] * 64)
                if abs(value) >= 100:
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


def print_spsa_params_all(model):
    c_end_weights = 6
    c_end_biases = 128

    num_weights = 0
    num_biases = 0

    stack_range = range(8)

    # feature transformer biases - 3072
    for j in range(3072):
        value = int(model.input.bias[j] * 254)
        print(f"ftB[{j}],{value},-1024,1024,16,0.0020")

    # L1 biases - 8 x 16 = 128
    for j in range(128):
        value = int(model.layer_stacks.l1.bias[j] * 64 * 127)
        print(f"oneB[{j}],{value},-20000,20000,{c_end_biases},0.0020")
        num_biases += 1

    # L2 weights - 8 x 960 = 7680
    for i in stack_range:
        for j in range(32):
            for k in range(30):
                value = int(model.layer_stacks.l2.weight[32 * i + j, k] * 64)
                print(f"twoW[{i}][{j}][{k}],{value},-127,127,{c_end_weights},0.0020")
                num_weights += 1

    # L2 biases - 8 x 32 = 256
    for i in stack_range:
        for j in range(32):
            value = int(model.layer_stacks.l2.bias[32 * i + j] * 64 * 127)
            print(f"twoB[{i}][{j}],{value},-20000,20000,{c_end_biases},0.0020")
            num_weights += 1

    # output weights - 8 x 32 = 256
    for i in stack_range:
        for j in range(32):
            value = round(int(model.layer_stacks.output.weight[i, j] * 600 * 16) / 127)
            print(f"oW[{i}][{j}],{value},-127,127,{c_end_weights},0.0020")
            num_weights += 1

    # output biases - 8
    for j in range(8):
        value = int(model.layer_stacks.output.bias[j] * 600 * 16)
        print(f"oB[{j}],{value},-20000,20000,{c_end_biases},0.0020")
        num_biases += 1

    print(f"# weights to tune: {num_weights}")
    print(f"# biases to tune:  {num_biases}")


def prep_spsa_params(nnue_filename):
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        reader = NNUEReader(f, feature_set)
        model = reader.model
    # prep_l2_weights_stack0(model)
    # prep_l2_weights(model)
    # prep_ft_biases(model)
    print_spsa_params_all(model)


if __name__ == "__main__":
    prep_spsa_params("nnue/nn-ddcfb9224cdb.nnue")
