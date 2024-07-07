import hashlib
import os
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import requests

import features
from serialize import NNUEReader, NNUEWriter


def print_spsa(spsa_page_url):
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

    for row in params_rows:
        td = row.find_all("td")
        param_name = td[0].text.strip()
        start_value = int(td[2].text)
        print(f"{param_name},{start_value}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 spsa_to_txt.py <spsa_page_url>")
        sys.exit(0)

    spsa_page_url = sys.argv[1]
    print_spsa(spsa_page_url)
