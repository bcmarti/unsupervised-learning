import os
import re
from collections import defaultdict

dataset_path = "3W/dataset"

well_counts = defaultdict(int)

for root, dirs, files in os.walk(dataset_path):
    for filename in files:
        if filename.startswith("WELL"):
            match = re.match(r"WELL-(\d+)", filename)
            if match:
                well_number = int(match.group(1))
                well_counts[well_number] += 1

if not well_counts:
    print("No WELL files found.")
else:
    print(f"{'Well':<10} {'Count':>6}")
    print("-" * 18)
    for well_num in sorted(well_counts):
        print(f"WELL-{well_num:05d}  {well_counts[well_num]:>6}")
    print("-" * 18)
    print(f"{'TOTAL':<10} {sum(well_counts.values()):>6}")