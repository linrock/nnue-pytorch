import hashlib
import os
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import requests

import features
from serialize import NNUEReader, NNUEWriter


def prep_spsa_params(nnue_filename):
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        reader = NNUEReader(f, feature_set)
        model = reader.model

    c_end = 16
    for i,value in enumerate(model.input.bias.data[:3072]):
        value_int = int(value * 254)
        if value_int < -250 or value_int > 250:
            print(f"ftB[{i}],{value_int},-1024,1024,{c_end},0.0020")


if __name__ == "__main__":
    prep_spsa_params("nn-ddcfb9224cdb.nnue")
