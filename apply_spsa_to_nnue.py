#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import requests

import features
from serialize import NNUEReader, NNUEWriter


def get_sha256_hash(nnue_data):
    h = hashlib.sha256()
    h.update(nnue_data)
    return h.hexdigest()


def modify_nnue(nnue_filename, spsa_page_url):
    print(spsa_page_url)
    response = requests.get(spsa_page_url)
    soup = BeautifulSoup(response.text, "html.parser")

    num_games_played = None
    spsa_status_div = soup.find("div", {"class": "elo-results-top"})
    for row in spsa_status_div.text.strip().split("\n"):
        if row.strip():
            print(" ", row.strip())
            if "games played" in row:
                num_games_played = f"{int(int(row.split("/")[0]) / 1000)}k"
    print()

    spsa_params_table = soup.find_all("table")[1]
    params_rows = spsa_params_table.find_all("tr", class_="spsa-param-row")
    print(f"Found {len(params_rows)} spsa params")

    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        reader = NNUEReader(f, feature_set)
        model = reader.model

    # [not modified, modified]
    counts = {
        "ftB": [0, 0],
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

    for row in params_rows:
        td = row.find_all("td")

        entry = td[0].text.strip()
        entry_split = entry.replace("[", " ").replace("]", " ").split()
        entry_split[1:] = map(int, entry_split[1:])
        match len(entry_split):
            case 4: param_type, bucket, idx1, idx2 = entry_split
            case 3: param_type, bucket, idx = entry_split
            case 2: param_type, idx = entry_split

        value = float(td[1].text)
        start_value = int(td[2].text)

        if start_value == int(value):
            counts[param_type][0] += 1
            continue

        match param_type:
            case "ftB":
                change_magnitudes["biases"] += abs(int(model.input.bias.data[idx] * 254) - int(value))
                model.input.bias.data[idx] = value / 254
                counts[param_type][1] += 1

            case "oneB":
                change_magnitudes["biases"] += abs(
                    int(model.layer_stacks.l1.bias.data[idx] * 64 * 127) - int(value)
                )
                model.layer_stacks.l1.bias.data[idx] = value / (64 * 127)
                counts[param_type][1] += 1

            case "twoW":
                change_magnitudes["weights"] += abs(
                    int(model.layer_stacks.l2.weight.data[32 * bucket + idx1, idx2] * 64) - int(value)
                )
                model.layer_stacks.l2.weight.data[32 * bucket + idx1, idx2] = value / 64
                counts[param_type][1] += 1

            case "twoB":
                change_magnitudes["biases"] += abs(
                    int(model.layer_stacks.l2.bias.data[32 * bucket + idx] * 64 * 127) - int(value)
                )
                model.layer_stacks.l2.bias.data[32 * bucket + idx] = value / (64 * 127)
                counts[param_type][1] += 1

            case "oW":
                change_magnitudes["weights"] += abs(
                    round(int(model.layer_stacks.output.weight.data[bucket, idx] * 600 * 16) / 127) - int(value)
                )
                model.layer_stacks.output.weight.data[bucket, idx] = value / (600 * 16 / 127)
                counts[param_type][1] += 1

            case "oB":
                change_magnitudes["biases"] += abs(
                    int(model.layer_stacks.output.bias.data[idx] * 600 * 16) - int(value)
                )
                model.layer_stacks.output.bias.data[idx] = value / (600 * 16)
                counts[param_type][1] += 1

    for key in counts.keys():
        if any(counts[key]):
            print(f"  {key}:")
            print(f"    # params:      {sum(counts[key])}")
            print(f"    # modified:    {counts[key][1]}")
   
    print(f"magnitude of changes: weights {change_magnitudes['weights']}, biases {change_magnitudes['biases']}") 

    description = "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer."
    writer = NNUEWriter(model, description, ft_compression="leb128")

    sha256_nnue_output_filename = f"nn-{get_sha256_hash(writer.buf)[:12]}.nnue"

    if Path(sha256_nnue_output_filename).exists():
        print(f"{sha256_nnue_output_filename} already exists. doing nothing")
    else:
        print(f"saving modified nnue to {sha256_nnue_output_filename}")
        print(os.path.abspath(sha256_nnue_output_filename))
        with open(sha256_nnue_output_filename, "wb") as f:
              f.write(writer.buf)

    info = {
        "filepath": os.path.abspath(sha256_nnue_output_filename),
        "comment": f"{len(params_rows)} - {num_games_played}"
    }
    print(json.dumps(info))
    return sha256_nnue_output_filename


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # print("Usage: python3 modify_nnue.py <nnue_filename> <spsa_page_url>")
        print("Usage: python3 modify_nnue.py <spsa_page_url>")
        sys.exit(0)

    # nnue_filename = "nnue/nn-ddcfb9224cdb.nnue"
    nnue_filename = "nnue/nn-74f1d263ae9a.nnue"

    spsa_page_url = sys.argv[1]
    print(f"Modifying {nnue_filename.split('/')[-1]}")

    modify_nnue(nnue_filename, spsa_page_url)
