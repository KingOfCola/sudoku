import json

from counter.first_block_row import enumerate_canonical_first_band
from counter.collapser import collapse_by_permutation, collapser_by_column

def print_band(band):
    """Print a band (a (3, 9) array) in a human-readable format."""
    k = band.shape[0]
    for row in band:
        blocks = [row[b * k:(b + 1) * k] for b in range(k)]
        print(" ".join("".join(str(x) for x in block) for block in blocks))
    print()  # Add an empty line after each band

if __name__ == "__main__":
    count = 0
    bands = []

    for band in enumerate_canonical_first_band():
        count += 1
        bands.append(band)

    print(f"Total number of blocks generated: {count}")

    collapsed_bands_by_permutation = collapse_by_permutation(bands)
    print(f"Total number of blocks after collapsing by permutation: {len(collapsed_bands_by_permutation)}")

    with open("outputs/collapsed_bands.json", "w") as f:
        json.dump({str(k): v["orbit_size"] for k, v in collapsed_bands_by_permutation.items()}, f)

    collapsed_representatives = [v["representative"] for v in collapsed_bands_by_permutation.values()]
    collapsed_bands = list(collapser_by_column(collapsed_representatives).values())
    print(f"Total number of blocks after collapsing: {len(collapsed_bands)}")

    for i, band in enumerate(collapsed_bands[:10]):
        print(f"Band {i + 1}:")
        print_band(band)
