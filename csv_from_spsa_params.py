import sys


with open(sys.argv[1], "r") as f:
    spsa_params = f.read().strip()

csv_tokens = []
for row in spsa_params.split("\n"):
    row_split = row.split(",")
    csv_tokens.append(row_split[0])
    csv_tokens.append(row_split[1])

with open("spsa-params.csv", "w") as f:
    f.write(",".join(csv_tokens))
