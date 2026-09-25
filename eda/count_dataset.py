import os
import time

base_dir = r"student_resource/dataset"

files = [
    "train/train_source1.tsv",
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "train/train_ground_truth.tsv",
    "test/test_source1.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv",
]

print("=== FILE SIZES AND LINE COUNTS ===")
for rel_path in files:
    full_path = os.path.join(base_dir, rel_path)
    t0 = time.time()
    size_mb = os.path.getsize(full_path) / (1024 * 1024)
    line_count = 0
    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
        for line in f:
            if line.strip():
                line_count += 1
    t1 = time.time()
    print(f"{rel_path:<30} | {size_mb:8.2f} MB | {line_count:10,d} rows | Header: {header} | Time: {t1-t0:.2f}s")
