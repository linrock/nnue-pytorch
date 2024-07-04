import hashlib
import os
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import requests


def view_spsa_histogram(spsa_page_url):
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

    values = []
    start_values = []
    for row in params_rows:
        td = row.find_all("td")
        param_name = td[0].text.strip()

        value = float(td[1].text)
        start_value = int(td[2].text)

        values.append(value)
        start_values.append(start_value)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    axes[0].hist(start_values, bins=30, alpha=0.75, color='green', edgecolor='black')
    axes[0].set_title('spsa start values')
    axes[0].set_xlabel('value')
    axes[0].set_ylabel('frequency')

    axes[1].hist(values, bins=30, alpha=0.75, color='orange', edgecolor='black')
    axes[1].set_title('spsa values')
    axes[1].set_xlabel('value')
    axes[1].set_ylabel('frequency')

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 spsa_hist.py <spsa_page_url>")
        sys.exit(0)

    spsa_page_url = sys.argv[1]
    view_spsa_histogram(spsa_page_url)
