import hashlib
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

    spsa_status_div = soup.find("div", {"class": "elo-results-top"})
    for row in spsa_status_div.text.strip().split("\n"):
        if row.strip():
            print(" ", row.strip())
    print()

    spsa_params_table = soup.find_all("table")[1]
    params_rows = spsa_params_table.find_all("tr", class_="spsa-param-row")
    print(f"Found {len(params_rows)} spsa params")

    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        reader = NNUEReader(f, feature_set)
        model = reader.model

    num_ft_b = 0
    num_ft_b_modified = 0
    ft_b_mod_magnitude = 0

    num_l2_w = 0
    num_l2_w_modified = 0
    l2_w_mod_magnitude = 0

    for row in params_rows:
        td = row.find_all("td")
        param_name = td[0].text.strip()

        value = float(td[1].text)
        start_value = int(td[2].text)

        if param_name.startswith("twoW"):
            num_l2_w += 1
            param_type, bucket, idx1, idx2 = param_name.replace("[", " ").replace("]", " ").split()
            i1 = 32 * int(bucket) + int(idx1)
            i2 = int(idx2)
            if int(model.layer_stacks.l2.weight[i1, i2] * 64) != start_value:
                print(f"warning: model.layer_stacks.l2.weight[{i1}, {i2}] != {start_value}")
            if round(value) != start_value:
                num_l2_w_modified += 1
                model.layer_stacks.l2.weight.data[i1, i2] = value / 64
                l2_w_mod_magnitude += abs(round(value) - start_value)

        if param_name.startswith("ftB"):
            num_ft_b += 1
            param_type, idx = param_name.replace("[", " ").replace("]", " ").split()
            if int(model.input.bias[int(idx)] * 254) != start_value:
                print(f"warning: model.input.bias[{int(idx)}] != {start_value}")
            if round(value) != start_value:
                num_ft_b_modified += 1
                model.input.bias.data[int(idx)] = value / 254
                ft_b_mod_magnitude += abs(round(value) - start_value)

    if num_ft_b > 0:
        print( "  FT bias:")
        print(f"    # params:      {num_ft_b}")
        print(f"    # modified:    {num_ft_b_modified}")
        print(f"    mod magnitude: {ft_b_mod_magnitude}")
        print()

    if num_l2_w > 0:
        print( "  L2 weights:")
        print(f"    # params:      {num_l2_w}")
        print(f"    # modified:    {num_l2_w_modified}")
        print(f"    mod magnitude: {l2_w_mod_magnitude}")
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

    return sha256_nnue_output_filename


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 modify_nnue.py <nnue_filename> <spsa_page_url>")
        sys.exit(0)

    nnue_filename = os.path.abspath(sys.argv[1])
    spsa_page_url = sys.argv[2]

    modify_nnue(nnue_filename, spsa_page_url)
