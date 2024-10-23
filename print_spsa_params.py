import hashlib
import os
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import requests

import features
from serialize import NNUEReader, NNUEWriter


class NnueSpsaParamPrinter(object):
    def __init__(self, nnue_filename):
        feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
        with open(nnue_filename, "rb") as f:
            reader = NNUEReader(f, feature_set)
        self.model = reader.model
        self.c_end_weights = 6
        self.c_end_biases = 128
        self.stack_range = range(8)

    # FT biases - 3072
    def print_ft_biases(self, condition=lambda value: True):
        c_end = 16
        for i,value in enumerate(self.model.input.bias.data[:3072]):
            value_int = int(value * 255)
            if condition(value_int):
                print(f"ftB[{i}],{value_int},-1024,1024,{c_end},0.0020")

    # FT weights - 8 x 16 x 3072 = 393,216
    def print_l1_weights(self, condition=lambda value: True):
        for i in self.stack_range:
            for j in range(16):
                for k in range(3072):
                    value = int(self.model.layer_stacks.l1.weight[16 * i + j, k] * 64)
                    if condition(value):
                        print(f"oneW[{i}][{j}][{k}],{value},-127,127,{self.c_end_weights},0.0020")

    # L1 biases - 8 x 16 = 128
    def print_l1_biases(self):
        for j in range(128):
            value = int(self.model.layer_stacks.l1.bias[j] * 64 * 127)
            print(f"oneB[{j}],{value},-16384,16384,{self.c_end_biases},0.0020")

    # L2 weights - 8 x 32 x 30 = 7680
    def print_l2_weights(self, condition=lambda value: True):
        for i in self.stage_range:
            for j in range(32):
                for k in range(30):
                    value = int(self.model.layer_stacks.l2.weight[32 * i + j, k] * 64)
                    if condition(value):
                        print(f"twoW[{i}][{j}][{k}],{value},-127,127,{self.c_end_weights},0.0020")

    # L2 biases - 8 x 32 = 256
    def print_l2_biases(self):
        for i in self.stack_range:
            for j in range(32):
                value = int(self.model.layer_stacks.l2.bias[32 * i + j] * 64 * 127)
                print(f"twoB[{i}][{j}],{value},-16384,16384,{self.c_end_biases},0.0020")

    # Output weights - 8 x 32 = 256
    def print_output_weights(self):
        for i in self.stack_range:
            for j in range(32):
                value = round(int(self.model.layer_stacks.output.weight[i, j] * 600 * 16) / 127)
                print(f"oW[{i}][{j}],{value},-127,127,{self.c_end_weights},0.0020")

    # output biases - 8
    def print_output_biases(self):
        for j in self.stack_range:
            value = int(self.model.layer_stacks.output.bias[j] * 600 * 16)
            print(f"oB[{j}],{value},-8192,8192,{self.c_end_biases},0.0020")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 print_spsa_params <nnue_filepath>")
        sys.exit(0)

    n = NnueSpsaParamPrinter(sys.argv[1])
    # print_ft_biases(model, lambda value: abs(value) < 50)

    # print_l1_weights(model, lambda value: abs(value) == 0)
    # print_l1_biases(model)

    # print_l2_weights(model, lambda value: abs(value) > 50)
    # print_l2_biases(model)

    n.print_output_weights()
    n.print_output_biases()
