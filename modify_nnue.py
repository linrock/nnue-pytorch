import hashlib
import os
from pathlib import Path
import sys

import features
from serialize import NNUEReader, NNUEWriter


def get_sha256_hash(nnue_data):
    h = hashlib.sha256()
    h.update(nnue_data)
    return h.hexdigest()


def modify_nnue(nnue_filename, spsa_csv_filename):
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        reader = NNUEReader(f, feature_set)
        model = reader.model

    # [not modified, modified]
    counts = {
        "ftW": [0, 0],
        "ftB": [0, 0],
        "oneW": [0, 0],
        "oneB": [0, 0],
        "twoW": [0, 0],
        "twoB": [0, 0],
        "oW": [0, 0],
        "oB": [0, 0],
    }
    change_magnitudes = {
        "weights": 0,
        "biases": 0
    }

    # ftB[0],-120,ftB[1],102,ftB[2],-112, ...
    # twoW[3][0][0],-26,twoW[3][0][1],-36,twoW[3][0][2],-60, ...
    with open(spsa_csv_filename, "r") as f:
        csv_stream = f.read().strip().split(",")
        param_type = None
        for entry in csv_stream:
            entry_split = entry.replace("[", " ").replace("]", " ").split()
            entry_split[1:] = map(int, entry_split[1:])
            match len(entry_split):
                case 4: param_type, bucket, idx1, idx2 = entry_split
                case 3: param_type, bucket, idx = entry_split
                case 2: param_type, idx = entry_split
                case 1:
                    value = float(entry_split[0])
                    match param_type:
                        # todo: double-check. hard to tune [-1, 0, 1] anyways
                        case "ftW":
                            if int(model.input.weight.data[idx] * 254) == int(value):
                                counts[param_type][0] += 1
                            else:
                                change_magnitudes["weights"] += abs(int(model.input.weight.data[idx] * 254) - int(value))
                                model.input.bias.data[idx] = value
                                counts[param_type][1] += 1

                        case "ftB":
                            if int(model.input.bias.data[idx] * 255) == int(value):
                                counts[param_type][0] += 1
                            else:
                                change_magnitudes["biases"] += abs(int(model.input.bias.data[idx] * 255) - int(value))
                                model.input.bias.data[idx] = value / 255
                                counts[param_type][1] += 1

                        case "oneW":
                            if int(model.layer_stacks.l1.weight.data[16 * bucket + idx1, idx2] * 64) == int(value):
                                counts[param_type][0] += 1
                            else:
                                change_magnitudes["weights"] += abs(
                                    int(model.layer_stacks.l1.weight.data[16 * bucket + idx1, idx2]) * 64 - int(value)
                                )
                                model.layer_stacks.l1.weight.data[16 * bucket + idx1, idx2] = value / 64
                                counts[param_type][1] += 1

                        case "oneB":
                            if int(model.layer_stacks.l1.bias.data[idx] * 64 * 127) == int(value):
                                counts[param_type][0] += 1
                            else:
                                change_magnitudes["biases"] += abs(
                                    int(model.layer_stacks.l1.bias.data[idx] * 64 * 127) - int(value)
                                )
                                model.layer_stacks.l1.bias.data[idx] = value / (64 * 127)
                                counts[param_type][1] += 1

                        case "twoW":
                            if int(model.layer_stacks.l2.weight.data[32 * bucket + idx1, idx2] * 64) == int(value):
                                counts[param_type][0] += 1
                            else:
                                change_magnitudes["weights"] += abs(
                                    int(model.layer_stacks.l2.weight.data[32 * bucket + idx1, idx2] * 64) - int(value)
                                )
                                model.layer_stacks.l2.weight.data[32 * bucket + idx1, idx2] = value / 64
                                counts[param_type][1] += 1

                        case "twoB":
                            if int(model.layer_stacks.l2.bias.data[32 * bucket + idx] * (64 * 127)) == int(value):
                                counts[param_type][0] += 1
                            else:
                                change_magnitudes["biases"] += abs(
                                    int(model.layer_stacks.l2.bias.data[32 * bucket + idx] * 64 * 127) - int(value)
                                )
                                model.layer_stacks.l2.bias.data[32 * bucket + idx] = value / (64 * 127)
                                counts[param_type][1] += 1

                        case "oW":
                            if round(int(model.layer_stacks.output.weight.data[bucket, idx] * 600 * 16) / 127) == int(value):
                                counts[param_type][0] += 1
                            else:
                                change_magnitudes["weights"] += abs(
                                    round(int(model.layer_stacks.output.weight.data[bucket, idx] * 600 * 16) / 127) - int(value)
                                )
                                model.layer_stacks.output.weight.data[bucket, idx] = value / (600 * 16 / 127)
                                counts[param_type][1] += 1

                        case "oB":
                            if int(model.layer_stacks.output.bias.data[idx] * 600 * 16) == int(value):
                                counts[param_type][0] += 1
                            else:
                                change_magnitudes["biases"] += abs(
                                    int(model.layer_stacks.output.bias.data[idx] * 600 * 16) - int(value)
                                )
                                model.layer_stacks.output.bias.data[idx] = value / (600 * 16)
                                counts[param_type][1] += 1

    if any(counts["ftW"]):
        print(f"# FT weights:     {counts['ftW'][0]} not modified, {counts['ftW'][1]} modified")

    if any(counts["ftB"]):
        print(f"# FT biases:      {counts['ftB'][0]} not modified, {counts['ftB'][1]} modified")

    if any(counts["oneW"]):
        print(f"# L1 weights:     {counts['oneW'][0]} not modified, {counts['oneW'][1]} modified")

    if any(counts["oneB"]):
        print(f"# L1 biases:      {counts['oneB'][0]} not modified, {counts['oneB'][1]} modified")

    if any(counts["twoW"]):
        print(f"# L2 weights:     {counts['twoW'][0]} not modified, {counts['twoW'][1]} modified")

    if any(counts["twoB"]):
        print(f"# L2 biases:      {counts['twoB'][0]} not modified, {counts['twoB'][1]} modified")

    if any(counts["oW"]):
        print(f"# output weights: {counts['oW'][0]} not modified, {counts['oW'][1]} modified")

    if any(counts["oB"]):
        print(f"# output biases:  {counts['oB'][0]} not modified, {counts['oB'][1]} modified")

    print(f"magnitude of changes: weights {change_magnitudes['weights']}, biases {change_magnitudes['biases']}")

    description = "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer."
    writer = NNUEWriter(model, description, ft_compression="leb128")

    sha256_nnue_output_filename = f"nn-{get_sha256_hash(writer.buf)[:12]}.nnue"

    if Path(sha256_nnue_output_filename).exists():
        print(f"{sha256_nnue_output_filename} already exists. doing nothing")
    else:
        print(f"saving modified nnue to {sha256_nnue_output_filename}")
        with open(sha256_nnue_output_filename, "wb") as f:
              f.write(writer.buf)

    return sha256_nnue_output_filename


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 modify_nnue.py <nnue_filename> <spsa_csv>")
        sys.exit(0)

    nnue_filename = os.path.abspath(sys.argv[1])
    spsa_csv_filename = os.path.abspath(sys.argv[2])
    modify_nnue(nnue_filename, spsa_csv_filename)
