import numpy as np
from counter.first_block_row import enumerate_blocks_first_rb

if __name__ == "__main__":
    count = 0
    
    for block in enumerate_blocks_first_rb():
        # print("".join(block.astype(str).tolist()))
        count += 1

    print(f"Total number of blocks generated: {count}")