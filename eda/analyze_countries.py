import os
import time
from collections import Counter

train_dir = r"student_resource/dataset/train"
test_dir = r"student_resource/dataset/test"

print("=== 1. ANALYZING COUNTRY DISTRIBUTIONS ===")

def get_country_counts(tsv_path):
    t0 = time.time()
    counts = Counter()
    total = 0
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        country_idx = header.index("country")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) > country_idx:
                c = parts[country_idx].strip()
                counts[c] += 1
            else:
                counts["MISSING"] += 1
            total += 1
    t1 = time.time()
    return counts, total, t1 - t0

datasets = [
    ("train/train_source1.tsv", os.path.join(train_dir, "train_source1.tsv")),
    ("train/train_source2.tsv", os.path.join(train_dir, "train_source2.tsv")),
    ("train/train_source3.tsv", os.path.join(train_dir, "train_source3.tsv")),
    ("test/test_source1.tsv", os.path.join(test_dir, "test_source1.tsv")),
    ("test/test_source2.tsv", os.path.join(test_dir, "test_source2.tsv")),
    ("test/test_source3.tsv", os.path.join(test_dir, "test_source3.tsv")),
]

for label, p in datasets:
    counts, total, duration = get_country_counts(p)
    breakdown = ", ".join([f"{k}: {v:,d} ({v/total*100:.2f}%)" for k, v in counts.most_common()])
    print(f"{label:<25} | Total: {total:10,d} | {breakdown} | in {duration:.2f}s")

print("\n=== 2. CHECKING CROSS-COUNTRY MATCHES IN GROUND TRUTH ===")
# Load country mapping for S1 (2.2M entries) and S2/S3
# To keep memory modest, store country as 1 byte / small integer or dict
print("Loading S1 countries...")
s1_country = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            s1_country[parts[0]] = parts[3]

print(f"Loaded {len(s1_country):,d} S1 countries.")

print("Loading S2 and S3 countries for matched entities only...")
# First let's read ground truth to see which S2 and S3 are actually matched
gt_pairs = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2 and parts[1].strip():
            s1_id = parts[0]
            matches = [m.strip() for m in parts[1].split(",") if m.strip()]
            for m in matches:
                gt_pairs.append((s1_id, m))

print(f"Total matched pairs in GT: {len(gt_pairs):,d}")
target_s2_needed = {m for s1, m in gt_pairs if m.startswith("S2-")}
target_s3_needed = {m for s1, m in gt_pairs if m.startswith("S3-")}

s2_country = {}
print("Loading S2 countries for matched targets...")
with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4 and parts[0] in target_s2_needed:
            s2_country[parts[0]] = parts[3]

s3_country = {}
print("Loading S3 countries for matched targets...")
with open(os.path.join(train_dir, "train_source3.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4 and parts[0] in target_s3_needed:
            s3_country[parts[0]] = parts[3]

print("Checking country consistency across all 7.6M matched pairs...")
same_country = 0
diff_country = 0
diff_examples = []

for s1_id, match_id in gt_pairs:
    c1 = s1_country.get(s1_id)
    if match_id.startswith("S2-"):
        c2 = s2_country.get(match_id)
    else:
        c2 = s3_country.get(match_id)
    
    if c1 == c2:
        same_country += 1
    else:
        diff_country += 1
        if len(diff_examples) < 10:
            diff_examples.append((s1_id, c1, match_id, c2))

print(f"Same country matches: {same_country:,d} ({same_country / len(gt_pairs) * 100:.4f}%)")
print(f"Cross country matches: {diff_country:,d} ({diff_country / len(gt_pairs) * 100:.4f}%)")
if diff_examples:
    print("Examples of cross-country matches:", diff_examples)
else:
    print("PROVEN: Zero cross-country matches! Country is a 100% strict partition!")
