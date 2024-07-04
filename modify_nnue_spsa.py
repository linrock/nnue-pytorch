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

    spsa_status_el = soup.find("div", {"class": "elo-results-top"})
    for row in spsa_status_el.text.strip().split("\n"):
        if row.strip():
            print(" ", row.strip())

    spsa_params_table = soup.find_all("table")[1]
    params_rows = spsa_params_table.find_all("tr", class_="spsa-param-row")
    print(f"Found {len(params_rows)} spsa params")

    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        reader = NNUEReader(f, feature_set)
        model = reader.model

    for row in params_rows:
        td = row.find_all("td")
        param_name = td[0].text.strip()
        value = float(td[1].text)

        if param_name.startswith("twoW"):
            param_type, bucket, idx1, idx2 = param_name.replace("[", " ").replace("]", " ").split()
            model.layer_stacks.l2.weight.data[32*int(bucket) + int(idx1), int(idx2)] = value / 64

        elif param_name.startswith("ftB"):
            param_type, idx = param_name.replace("[", " ").replace("]", " ").split()
            model.input.bias.data[int(idx)] = value / 254

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
        print("Usage: python3 modify_nnue.py <nnue_filename> <spsa_page_url>")
        sys.exit(0)

    nnue_filename = os.path.abspath(sys.argv[1])
    spsa_page_url = sys.argv[2]

    modify_nnue(nnue_filename, spsa_page_url)
