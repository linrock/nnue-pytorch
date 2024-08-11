import hashlib
import os
from pathlib import Path
import sys

import features
from serialize import NNUEReader, NNUEWriter


feature_set = features.get_feature_set_from_name("HalfKAv2_hm")


def get_sha256_hash(nnue_data):
    h = hashlib.sha256()
    h.update(nnue_data)
    return h.hexdigest()


def combine_nnue(apply_nnues):
    # base_nnue = "nnue/nn-31337bea577c.nnue"
    base_nnue = apply_nnues[0]
    print(f"base nnue: {base_nnue}")
    with open(base_nnue, "rb") as f:
        base_model = NNUEReader(f, feature_set).model

    apply_nnues = apply_nnues[1:]

    # [not modified, modified]
    counts = {
        "ftB": [0, 0],
        "oneB": [0, 0],
        "twoW": [0, 0],
        "twoB": [0, 0],
        "oW": [0, 0],
        "oB": [0, 0],
    }
    changed_params = {
        "ftB": set(),
        "twoW": set(),
        "oW": set(),
        "oB": set(),
    }
    change_magnitudes = {
        "weights": 0,
        "biases": 0
    }

    stack_range = range(8)

    for apply_nnue in apply_nnues:
        print(f"apply nnue: {apply_nnue}")
        with open(apply_nnue, "rb") as f:
            apply_model = NNUEReader(f, feature_set).model

        param_type = "ftB"
        for j in range(3072):
            if j in changed_params[param_type]:
                pass
            elif base_model.input.bias.data[j] == apply_model.input.bias[j]:
                counts[param_type][0] += 1
            else:
                base_model.input.bias.data[j] = apply_model.input.bias[j]
                counts[param_type][1] += 1
                changed_params[param_type].add(j)
     
        # L2 weights - 8 x 960 = 7680
        param_type = "twoW" 
        for i in stack_range:
            for j in range(32):
                for k in range(30):
                    key = f"{i},{j},{k}" 
                    if key in changed_params[param_type]:
                        pass
                    elif base_model.layer_stacks.l2.weight[32 * i + j, k] == apply_model.layer_stacks.l2.weight[32 * i + j, k]:
                        counts[param_type][0] += 1
                    else:
                        base_model.layer_stacks.l2.weight.data[32 * i + j, k] = apply_model.layer_stacks.l2.weight[32 * i + j, k]
                        counts[param_type][1] += 1
                        changed_params[param_type].add(key)

        # output weights - 8 x 960 = 7680
        param_type = "oW" 
        for i in stack_range:
            for j in range(30):
                key = f"{i},{j}" 
                if key in changed_params[param_type]:
                    pass
                elif base_model.layer_stacks.output.weight[i, j] == apply_model.layer_stacks.output.weight[i, j]:
                    counts[param_type][0] += 1
                else:
                    base_model.layer_stacks.output.weight.data[i, j] = apply_model.layer_stacks.output.weight[i, j]
                    counts[param_type][1] += 1
                    changed_params[param_type].add(key)

        # L2 weights - 8 x 960 = 7680
        param_type = "oB" 
        for i in stack_range:
            if i in changed_params[param_type]:
                pass
            elif base_model.layer_stacks.output.bias[i] == apply_model.layer_stacks.output.bias[i]:
                counts[param_type][0] += 1
            else:
                base_model.layer_stacks.output.bias.data[i] = apply_model.layer_stacks.output.bias[i]
                counts[param_type][1] += 1
                changed_params[param_type].add(i)

        if any(counts["ftB"]):
            print(f"# FT biases:      {counts['ftB'][0]} not modified, {counts['ftB'][1]} modified")

        if any(counts["twoW"]):
            print(f"# L2 weights:     {counts['twoW'][0]} not modified, {counts['twoW'][1]} modified")

        if any(counts["oW"]):
            print(f"# output weights: {counts['oW'][0]} not modified, {counts['oW'][1]} modified")

        if any(counts["oB"]):
            print(f"# output biases:  {counts['oB'][0]} not modified, {counts['oB'][1]} modified")

    # print(f"magnitude of changes: weights {change_magnitudes['weights']}, biases {change_magnitudes['biases']}")

    description = "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer."
    writer = NNUEWriter(base_model, description, ft_compression="leb128")

    sha256_nnue_output_filename = f"nn-{get_sha256_hash(writer.buf)[:12]}.nnue"

    if Path(sha256_nnue_output_filename).exists():
        print(f"{sha256_nnue_output_filename} already exists. doing nothing")
    else:
        print(f"saving modified nnue to {sha256_nnue_output_filename}")
        with open(sha256_nnue_output_filename, "wb") as f:
              f.write(writer.buf)

    return sha256_nnue_output_filename


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 combine_nnue.py <base_nnue> <nnue2> <nnue3> ...")
        sys.exit(0)

    combine_nnue(sys.argv[1:])
