from pprint import pprint
import sys

import numpy as np

import features
from serialize import NNUEReader, NNUEWriter


def print_changes(filename1, filename2, print_spsa_params):
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(filename1, "rb") as f:
        nnue1 = NNUEReader(f, feature_set).model
    with open(filename2, "rb") as f:
        nnue2 = NNUEReader(f, feature_set).model

    num_weights = 0
    weight_diffs = []

    num_biases = 0

    stack_range = range(8)

    # feature transformer biases - 3072
    for j in range(3072):
        value1 = int(nnue1.input.bias[j] * 254)
        value2 = int(nnue2.input.bias[j] * 254)
        if value1 != value2:
            print(f"ftB[{j}] {value1} -> {value2}")
            num_biases += 1

    # L1 biases - 8 x 16 = 128
    for j in range(128):
        value1 = int(nnue1.layer_stacks.l1.bias[j] * 64 * 127)
        value2 = int(nnue2.layer_stacks.l1.bias[j] * 64 * 127)
        if value1 != value2:
            print(f"oneB[{j}] {value1} -> {value2}")
            num_biases += 1

    # L2 weights - 8 x 960 = 7680
    changes_by_bucket = {}
    for i in stack_range:
        for j in range(32):
            for k in range(30):
                value1 = int(nnue1.layer_stacks.l2.weight[32 * i + j, k] * 64)
                value2 = int(nnue2.layer_stacks.l2.weight[32 * i + j, k] * 64)
                if value1 != value2:
                    print(f"twoW[{i}][{j}][{k}] {value1} -> {value2}")
                    num_weights += 1
                    if not changes_by_bucket.get(i):
                        changes_by_bucket[i] = { "count": 0, "diffs": [] }
                    changes_by_bucket[i]["count"] += 1
                    diff = value2 - value1
                    changes_by_bucket[i]["diffs"].append(diff)
                    weight_diffs.append(diff)

    # L2 biases - 8 x 32 = 256
    for i in stack_range:
        for j in range(32):
            value1 = int(nnue1.layer_stacks.l2.bias[32 * i + j] * 64 * 127)
            value2 = int(nnue2.layer_stacks.l2.bias[32 * i + j] * 64 * 127)
            if value1 != value2:
                print(f"twoB[{i}][{j}] {value1} -> {value2}")
                num_weights += 1

    # output weights - 8 x 32 = 256
    output_changes_by_bucket = {}
    output_weight_diffs = []
    num_output_weights = 0

    c_end_weights = 6
    c_end_biases = 32

    for i in stack_range:
        for j in range(32):
            value1 = round(int(nnue1.layer_stacks.output.weight[i, j] * 600 * 16) / 127)
            value2 = round(int(nnue2.layer_stacks.output.weight[i, j] * 600 * 16) / 127)
            if value1 != value2:
                if print_spsa_params:
                    print(f"oW[{i}][{j}],{value2},-127,127,{c_end_weights},0.0020")
                else:
                    print(f"oW[{i}][{j}] {value1} -> {value2}")
                num_output_weights += 1
                diff = value2 - value1
                if not output_changes_by_bucket.get(i):
                    output_changes_by_bucket[i] = { "count": 0, "diffs": [] }
                output_changes_by_bucket[i]["count"] += 1
                diff = value2 - value1
                output_changes_by_bucket[i]["diffs"].append(diff)
                output_weight_diffs.append(diff)

    # output biases - 8
    for j in range(8):
        value1 = int(nnue1.layer_stacks.output.bias[j] * 600 * 16)
        value2 = int(nnue2.layer_stacks.output.bias[j] * 600 * 16)
        if value1 != value2:
            if print_spsa_params:
                print(f"oB[{j}],{value2},-4096,4096,{c_end_biases},0.0020")
            else:
                print(f"oB[{j}] {value1} -> {value2}")
            num_biases += 1

    if not print_spsa_params:
        print(f"{filename1} -> {filename2}")
        if weight_diffs:
            print(f"# L2 weights changed: {num_weights}")
            print(f"  avg: {np.mean(weight_diffs):.4f} +/- {np.std(weight_diffs):.4f}")
            print(f"  min: {min(weight_diffs)}")
            print(f"  max: {max(weight_diffs)}")
        if num_output_weights:
            print(f"# output weights changed: {num_output_weights}")
            print(f"  avg: {np.mean(output_weight_diffs):.4f} +/- {np.std(output_weight_diffs):.4f}")
            print(f"  min: {min(output_weight_diffs)}")
            print(f"  max: {max(output_weight_diffs)}")
        for bucket, stats in changes_by_bucket.items():
            total_change = sum([abs(d) for d in stats["diffs"]])
            print(f"{bucket}: {stats['count']} changed, magnitude: {total_change}")
        # print(changes_by_bucket)
        print(f"# biases changed:  {num_biases}")


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python3 nnue_diff.py <nnue1> <nnue2> <spsa_params>")
        sys.exit(0)

    print_spsa_params = len(sys.argv) == 4
    print_changes(sys.argv[1], sys.argv[2], print_spsa_params)