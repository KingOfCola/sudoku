import json

if __name__ == "__main__":
    with open("outputs/collapsed_bands.json", "r") as f:
        collapsed_bands_by_permutation = json.load(f)

    print("total number of blocks after collapsing by permutation:", sum(v for v in collapsed_bands_by_permutation.values()))