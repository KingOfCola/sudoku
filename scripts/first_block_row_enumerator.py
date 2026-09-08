import numpy as np
from counter.first_block_row import enumerate_canonical_first_band
from counter.collapser import collapse_by_permutation, collapser_by_column

def print_band(band):
    """Print a band in a human-readable format."""
    for row in range(3):
        row_values = []
        for block in range(3):
            start = row * 9 + block * 3
            row_values.append(band[start:start + 3])
        print(" ".join(str("".join(str(x) for x in v)) for v in row_values))
    print()  # Add an empty line after each band

if __name__ == "__main__":
    count = 0
    bands = []
    
    for band in enumerate_canonical_first_band():
        # print("".join(block.astype(str).tolist()))
        count += 1
        bands.append(band)

    print(f"Total number of blocks generated: {count}")

    collapsed_bands_by_permutation = collapse_by_permutation(bands)
    print(f"Total number of blocks after collapsing by permutation: {len(collapsed_bands_by_permutation)}")

    collapsed_bands = list(collapser_by_column(collapsed_bands_by_permutation).values())
    print(f"Total number of blocks after collapsing: {len(collapsed_bands)}")

    for i, band in enumerate(collapsed_bands[:10]):
        print(f"Band {i + 1}:")
        print_band(band)