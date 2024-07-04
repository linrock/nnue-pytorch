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

    # twoW[3][0][0],-26,twoW[3][0][1],-36,twoW[3][0][2],-60, ...
    with open(spsa_csv_filename, "r") as f:
        csv_stream = f.read().strip().split(",")
        param_type = None
        for entry in csv_stream:
            if "twoW" in entry:
                param_type, bucket, idx1, idx2 = entry.replace("[", " ").replace("]", " ").split()
            elif "ftB" in entry:
                param_type, idx = entry.replace("[", " ").replace("]", " ").split()
            else:
                value = int(entry)
                if param_type == "twoW":
                    model.layer_stacks.l2.weight.data[32*int(bucket) + int(idx1), int(idx2)] = float(value) / 64
                elif param_type == "ftB":
                    model.input.bias.data[idx] = float(value) / 254

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
