import os
import json
from collections import defaultdict

data_root = "data/Image_data"
# regions = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and len(d) == 3 and d.isupper()]
# Allow POST (4 letters)
regions = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.isupper()]
classes = ["no_damage", "minor", "major", "destroyed", "affected"]

stats = defaultdict(lambda: defaultdict(int))
total_stats = defaultdict(int)

for region in regions:
    region_path = os.path.join(data_root, region, "train")
    if not os.path.exists(region_path):
        continue
    
    for cls in classes:
        cls_path = os.path.join(region_path, cls)
        if os.path.exists(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            stats[region][cls] = count
            total_stats[cls] += count

print("Total Stats:", dict(total_stats))
print("\nRegion Stats:")
for region, counts in stats.items():
    print(f"{region}: {dict(counts)}")

# Simple greedy split algorithm
# We want roughly 70/15/15 split of TOTAL samples, while keeping regions distinct.
# And we want to maintain class distribution if possible.

total_samples = sum(total_stats.values())
print(f"\nTotal Samples: {total_samples}")

# Calculate target ratios
target_ratios = {k: v / total_samples for k, v in total_stats.items()}
print("Target Class Ratios:", target_ratios)

import random
import numpy as np

best_split = None
best_score = float('inf')

# Try random splits
for _ in range(10000):
    shuffled_regions = list(stats.keys())
    random.shuffle(shuffled_regions)
    
    train_regions = []
    val_regions = []
    test_regions = []
    
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    train_size = 0
    val_size = 0
    test_size = 0
    
    # Greedy assignment based on size
    for r in shuffled_regions:
        r_counts = stats[r]
        r_total = sum(r_counts.values())
        
        # Assign to bucket that needs it most (based on size ratio)
        # Target: 0.7, 0.15, 0.15
        
        current_total = train_size + val_size + test_size + r_total
        if current_total == 0: continue
        
        train_ratio = train_size / current_total
        val_ratio = val_size / current_total
        test_ratio = test_size / current_total
        
        # Simple logic: fill up to target
        # But we want to balance classes too.
        # Let's just do random assignment with probability 0.7, 0.15, 0.15
        # and check score.
        
        rand = random.random()
        if rand < 0.7:
            train_regions.append(r)
            train_size += r_total
            for c, n in r_counts.items(): train_counts[c] += n
        elif rand < 0.85:
            val_regions.append(r)
            val_size += r_total
            for c, n in r_counts.items(): val_counts[c] += n
        else:
            test_regions.append(r)
            test_size += r_total
            for c, n in r_counts.items(): test_counts[c] += n

    if train_size == 0 or val_size == 0 or test_size == 0:
        continue

    # Check if all classes present
    if len(train_counts) < 5 or len(val_counts) < 5 or len(test_counts) < 5:
        # Penalize heavily if missing classes
        score = 1000
    else:
        score = 0
        
    # Calculate KL divergence or MSE of ratios
    for counts, size in [(train_counts, train_size), (val_counts, val_size), (test_counts, test_size)]:
        for cls in classes:
            ratio = counts[cls] / size
            target = target_ratios[cls]
            score += (ratio - target) ** 2
            
    # Size penalty
    total = train_size + val_size + test_size
    score += 10 * ((train_size/total - 0.7)**2 + (val_size/total - 0.15)**2 + (test_size/total - 0.15)**2)

    if score < best_score:
        best_score = score
        best_split = (train_regions, val_regions, test_regions)

print("\nBest Split Found:")
print("Train Regions:", best_split[0])
print("Val Regions:", best_split[1])
print("Test Regions:", best_split[2])

# Print stats for best split
for name, regions in zip(["Train", "Val", "Test"], best_split):
    print(f"\n{name} Stats:")
    counts = defaultdict(int)
    for r in regions:
        for c, n in stats[r].items():
            counts[c] += n
    total = sum(counts.values())
    print(f"Total: {total}")
    print(f"Ratios: {dict((k, round(v/total, 3)) for k, v in counts.items())}")


