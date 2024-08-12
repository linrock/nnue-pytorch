#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
from pprint import pprint
import sys

from bs4 import BeautifulSoup
import requests

import features
from serialize import NNUEReader, NNUEWriter


def get_sha256_hash(nnue_data):
    h = hashlib.sha256()
    h.update(nnue_data)
    return h.hexdigest()


def create_nnue_from_spsa_page(spsa_page_url):
    print(spsa_page_url)
    response = requests.get(spsa_page_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # print some info about the spsa test
    print()
    base_branch = soup.find("h2").find("span").text.strip().split()[-1]
    print(f"base branch: {base_branch}")

    # print progress of this spsa test
    num_games_played = None
    spsa_status_div = soup.find("div", {"class": "elo-results-top"})
    for row in spsa_status_div.text.strip().split("\n"):
        if row.strip():
            print(" ", row.strip())
            if "games played" in row:
                num_games_played = f"{int(int(row.split("/")[0]) / 1000)}k"
    print()

    # get the base net. spsa params will be applied to this net
    test_details_table = soup.find_all("table")[0]
    new_nets_main = None
    base_nets_main = None
    for tr in test_details_table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        if tds[0].text.strip() == "new_nets":
            new_nets_main = tds[1].text.strip().split(",")[0]
        elif tds[0].text.strip() == "base_nets":
            base_nets_main = tds[1].text.strip().split(",")[0]
        if new_nets_main and base_nets_main:
            if new_nets_main != base_nets_main:
                print(f"Expected {new_nets_main} to be the same as {base_nets_main}. Exiting")
                sys.exit(1)
            else:
                print(f"base net: {base_nets_main}")
                break

    # stats on the # of params
    nnue_filename = base_nets_main
    spsa_params_table = soup.find_all("table")[1]
    params_rows = spsa_params_table.find_all("tr", class_="spsa-param-row")

    use_latest_params = False
    param_history_index = -2

    # collect the latest params from the page
    params = []
    for row in params_rows:
        td = row.find_all("td")
        var_name = td[0].text.strip()
        value = float(td[1].text)
        start_value = int(td[2].text)
        params.append({
            "var_name": var_name,
            "start_value": start_value,
            "value": value,
        })

    # also get params from javascript spsaData var
    spsa_param_map = {}
    for script in soup.find_all("script"):
        if "spsaData" in script.text:
            spsa_data = json.loads(script.text.split("const spsaData = ")[-1].strip().strip(";"))
    param_names = [param["name"] for param in spsa_data["params"]]
    param_values = spsa_data["param_history"][param_history_index]
    param_history_length = len(spsa_data["param_history"])

    print("Latest params")
    for row in zip(param_names, param_values):
        spsa_param_map[row[0]] = round(row[1]["theta"])
    # pprint(spsa_param_map)

    if len(params) != len(spsa_param_map.keys()):
        print(f"size of params and spsa_param_map don't match: {len(params)} != {len(spsa_param_map.keys())}")
        sys.exit(1)

    print(f"Found {len(params_rows)} spsa params")

    # Use previous params from spsaData
    if not use_latest_params:
        print("Chosen params")
        for i,param in enumerate(params):
            param["value"] = spsa_param_map[param["var_name"]]
        # pprint(params)
        # sys.exit(0)

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

    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        reader = NNUEReader(f, feature_set)
        model = reader.model

    for param in params:
        entry = param["var_name"]
        entry_split = entry.replace("[", " ").replace("]", " ").split()
        entry_split[1:] = map(int, entry_split[1:])
        match len(entry_split):
            case 4: param_type, bucket, idx1, idx2 = entry_split
            case 3: param_type, bucket, idx = entry_split
            case 2: param_type, idx = entry_split

        value = param["value"]
        start_value = param["start_value"]

        if int(start_value) == int(value):
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

    param_types_changed = []
    num_biases_changed = 0
    num_weights_changed = 0
    for key in counts.keys():
        if any(counts[key]):
            print(f"  {key}:")
            print(f"    # params:      {sum(counts[key])}")
            print(f"    # modified:    {counts[key][1]}")
            param_types_changed.append(key)
        if key.endswith("W"):
            num_weights_changed += counts[key][1]
        elif key.endswith("B"):
            num_biases_changed += counts[key][1]

    print(f"# biases changed:   {num_biases_changed}")
    print(f"# weights changed:  {num_weights_changed}")
    print(f"magnitude of changes: weights {change_magnitudes['weights']}, biases {change_magnitudes['biases']}") 
    print()

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

    nnue_base = nnue_filename.split("nn-")[1][:6]
    changed_param_tokens = []
    if num_weights_changed > 0:
        changed_param_tokens.append(f"{num_weights_changed}W")
    if num_biases_changed > 0:
        changed_param_tokens.append(f"{num_biases_changed}B")
    if not use_latest_params:
        changed_param_tokens.append(f"{param_history_index}/{param_history_length}")

    change_str = " ".join(changed_param_tokens)

    info = {
        "base_branch": base_branch,
        "filepath": os.path.abspath(sha256_nnue_output_filename),
        "comment": f"{nnue_base}: {change_str}, {len(params_rows)} {" ".join(param_types_changed)} {num_games_played}"
    }
    print(json.dumps(info))
    return sha256_nnue_output_filename


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 apply_spsa_to_nnue.py <spsa_page_url>")
        sys.exit(0)

    # nnue_filename = "nnue/nn-ddcfb9224cdb.nnue"
    # nnue_filename = "nnue/nn-74f1d263ae9a.nnue"
    # nnue_filename = "nnue/nn-e8bac1c07a5a.nnue"
    # nnue_filename = "nnue/nn-31337bea577c.nnue"
    # print(f"Modifying {nnue_filename.split('/')[-1]} ...")

    spsa_page_url = sys.argv[1]
    create_nnue_from_spsa_page(spsa_page_url)
