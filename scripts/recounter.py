import json
from matplotlib import pyplot as plt
import numpy as np

from visualizer.display import show_grid

if __name__ == "__main__":
    # with open("outputs/collapsed_bands.json", "r") as f:
    #     collapsed_bands_by_permutation = json.load(f)

    # print("total number of blocks after collapsing by permutation:", sum(v for v in collapsed_bands_by_permutation.values()))

    grid = np.array([[0, 1, 2, 3],
                     [2, 3, 0, 1],])
    show_grid(grid)
    plt.show()