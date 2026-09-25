import os
import sys
import re
import random
import time
from collections import Counter
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

sys.stdout.reconfigure(encoding='utf-8')

train_dir = r"student_resource/dataset/train"
test_dir = r"student_resource/dataset/test"

print("=== DEEP TEXT, NOISE & SIMILARITY ANALYSIS ===")

# 1. Missing values & length statistics across sources
def inspect_field_stats(file_path, sample_limit=200000):
    t0 = time.time()
    name_lens = []
    addr_lens = []
    name_words = []
    addr_words = []
    empty_name = 0
    empty_addr = 0
    total = 0
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        name_idx = header.index("business_name")
        addr_idx = header.index("business_address")
        
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            total += 1
            if total > sample_limit:
                break
            
            name = parts[name_idx].strip() if len(parts) > name_idx else ""
            addr = parts[addr_idx].strip() if len(parts) > addr_idx else ""
            
            if not name:
                empty_name += 1
            else:
                name_lens.append(len(name))
                name_words.append(len(name.split()))
                
            if not addr:
                empty_addr += 1
            else:
                addr_lens.append(len(addr))
                addr_words.append(len(addr.split()))
                
    return {
        "total_sampled": total,
        "empty_name": empty_name,
        "empty_name_pct": empty_name / total * 100,
        "empty_addr": empty_addr,
        "empty_addr_pct": empty_addr / total * 100,
        "name_len_mean": np.mean(name_lens) if name_lens else 0,
        "name_len_median": np.median(name_lens) if name_lens else 0,
        "name_words_mean": np.mean(name_words) if name_words else 0,
        "name_words_median": np.median(name_words) if name_words else 0,
        "addr_len_mean": np.mean(addr_lens) if addr_lens else 0,
        "addr_len_median": np.median(addr_lens) if addr_lens else 0,
        "addr_words_mean": np.mean(addr_words) if addr_words else 0,
        "addr_words_median": np.median(addr_words) if addr_words else 0,
    }

files_to_check = [
    ("train_s1", os.path.join(train_dir, "train_source1.tsv")),
    ("train_s2", os.path.join(train_dir, "train_source2.tsv")),
    ("train_s3", os.path.join(train_dir, "train_source3.tsv")),
    ("test_s1", os.path.join(test_dir, "test_source1.tsv")),
    ("test_s2", os.path.join(test_dir, "test_source2.tsv")),
    ("test_s3", os.path.join(test_dir, "test_source3.tsv")),
]

print("\n--- Field Lengths & Null Rates (Sampled 200k rows each) ---")
for name, p in files_to_check:
    stats = inspect_field_stats(p, sample_limit=200000)
    print(f"{name:<10} | Null Name: {stats['empty_name_pct']:.2f}% | Null Addr: {stats['empty_addr_pct']:.2f}% | "
          f"Name Len (med/mean): {stats['name_len_median']:.0f}/{stats['name_len_mean']:.1f} ({stats['name_words_median']:.0f} words) | "
          f"Addr Len (med/mean): {stats['addr_len_median']:.0f}/{stats['addr_len_mean']:.1f} ({stats['addr_words_median']:.0f} words)")

# 2. Detailed Ground Truth Pair Inspection: Noise, String Similarities, Postal Codes
print("\n--- Sampling Ground Truth Pairs for Deep Noise & Similarity Analysis ---")
random.seed(42)

# Sample 50,000 ground truth rows
sampled_gt = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i % 44 == 0:  # ~50k samples out of 2.2M
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1].strip():
                s1_id = parts[0]
                matches = [m.strip() for m in parts[1].split(",") if m.strip()]
                for m in matches:
                    sampled_gt.append((s1_id, m))
            if len(sampled_gt) >= 60000:
                break

print(f"Sampled {len(sampled_gt):,d} true match pairs.")

needed_s1 = {s1 for s1, m in sampled_gt}
needed_s2 = {m for s1, m in sampled_gt if m.startswith("S2-")}
needed_s3 = {m for s1, m in sampled_gt if m.startswith("S3-")}

# Load S1 records
s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in needed_s1:
            s1_records[parts[0]] = {
                "name": parts[1] if len(parts) > 1 else "",
                "addr": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            }

# Load S2 records
s2_records = {}
with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8", errors="replace") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in needed_s2:
            s2_records[parts[0]] = {
                "name": parts[1] if len(parts) > 1 else "",
                "addr": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            }

# Load S3 records
s3_records = {}
with open(os.path.join(train_dir, "train_source3.tsv"), "r", encoding="utf-8", errors="replace") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in needed_s3:
            s3_records[parts[0]] = {
                "name": parts[1] if len(parts) > 1 else "",
                "addr": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            }

print(f"Loaded records: S1={len(s1_records):,d}, S2={len(s2_records):,d}, S3={len(s3_records):,d}")

# Postal code extractors
re_pin_in = re.compile(r'\b[1-9][0-9]{5}\b')
re_zip_us = re.compile(r'\b[0-9]{5}(?:-[0-9]{4})?\b')

def extract_postal_code(addr, country):
    if country == "India":
        m = re_pin_in.findall(addr)
        return m[-1] if m else None
    elif country == "US":
        m = re_zip_us.findall(addr)
        return m[-1][:5] if m else None
    return None

# Analyze true positive pairs
us_pairs = []
in_pairs = []

sim_results = []
examples_by_type = {
    "exact_name": [],
    "high_name_diff_addr": [],
    "low_name_high_addr": [],
    "abbrev_name": [],
    "address_reordered": []
}

for s1_id, match_id in sampled_gt:
    r1 = s1_records.get(s1_id)
    r2 = s2_records.get(match_id) if match_id.startswith("S2-") else s3_records.get(match_id)
    if not r1 or not r2:
        continue
    
    country = r1["country"]
    n1, n2 = r1["name"], r2["name"]
    a1, a2 = r1["addr"], r2["addr"]
    
    # Fast similarities
    ratio_name = fuzz.ratio(n1.lower(), n2.lower())
    token_sort_name = fuzz.token_sort_ratio(n1.lower(), n2.lower())
    token_set_name = fuzz.token_set_ratio(n1.lower(), n2.lower())
    
    ratio_addr = fuzz.ratio(a1.lower(), a2.lower())
    token_sort_addr = fuzz.token_sort_ratio(a1.lower(), a2.lower())
    token_set_addr = fuzz.token_set_ratio(a1.lower(), a2.lower())
    
    p1 = extract_postal_code(a1, country)
    p2 = extract_postal_code(a2, country)
    postal_status = "both_missing"
    if p1 and p2:
        postal_status = "match" if p1 == p2 else "mismatch"
    elif p1 or p2:
        postal_status = "one_missing"
        
    sim_results.append({
        "country": country,
        "source": match_id[:2],
        "name_ratio": ratio_name,
        "name_sort": token_sort_name,
        "name_set": token_set_name,
        "addr_ratio": ratio_addr,
        "addr_sort": token_sort_addr,
        "addr_set": token_set_addr,
        "postal_status": postal_status,
        "n1": n1, "n2": n2, "a1": a1, "a2": a2
    })
    
    # Collect qualitative examples
    if ratio_name == 100 and len(examples_by_type["exact_name"]) < 3:
        examples_by_type["exact_name"].append((n1, n2, a1, a2, country))
    elif token_set_name > 85 and ratio_name < 60 and len(examples_by_type["abbrev_name"]) < 3:
        examples_by_type["abbrev_name"].append((n1, n2, a1, a2, country))
    elif token_set_addr > 85 and ratio_addr < 50 and len(examples_by_type["address_reordered"]) < 3:
        examples_by_type["address_reordered"].append((n1, n2, a1, a2, country))
    elif token_sort_name < 50 and token_set_addr > 70 and len(examples_by_type["low_name_high_addr"]) < 3:
        examples_by_type["low_name_high_addr"].append((n1, n2, a1, a2, country))

df_sim = pd.DataFrame(sim_results)

print(f"\n--- TRUE MATCH SIMILARITY DISTRIBUTIONS (N={len(df_sim):,d}) ---")
print("Overall Metrics (Percentiles: 10%, 25%, 50%, 75%, 90%):")
for col in ["name_ratio", "name_sort", "name_set", "addr_ratio", "addr_sort", "addr_set"]:
    vals = df_sim[col].values
    p = np.percentile(vals, [10, 25, 50, 75, 90])
    mean_val = np.mean(vals)
    print(f"  {col:<12}: Mean={mean_val:5.1f} | P10={p[0]:4.1f} | P25={p[1]:4.1f} | Median={p[2]:4.1f} | P75={p[3]:4.1f} | P90={p[4]:4.1f}")

print("\n--- By Country Breakdown (Median Values) ---")
print(df_sim.groupby("country")[["name_ratio", "name_sort", "name_set", "addr_ratio", "addr_sort", "addr_set"]].median())

print("\n--- By Match Source Breakdown (Median Values) ---")
print(df_sim.groupby("source")[["name_ratio", "name_sort", "name_set", "addr_ratio", "addr_sort", "addr_set"]].median())

print("\n--- Postal Code Agreement in True Matches ---")
postal_counts = df_sim["postal_status"].value_counts()
for status, cnt in postal_counts.items():
    print(f"  {status:<15}: {cnt:6,d} ({cnt/len(df_sim)*100:5.2f}%)")

print("\n--- Postal Code by Country ---")
print(pd.crosstab(df_sim["country"], df_sim["postal_status"], normalize="index") * 100)

# Save df_sim summary to CSV for plotting
df_sim[["country", "source", "name_ratio", "name_sort", "name_set", "addr_ratio", "addr_sort", "addr_set", "postal_status"]].to_csv("eda/sim_sample_results.csv", index=False)
print("\nSaved eda/sim_sample_results.csv successfully.")

print("\n=== QUALITATIVE REAL-WORLD NOISE EXAMPLES FROM GROUND TRUTH ===")
for category, ex_list in examples_by_type.items():
    print(f"\n[{category.upper()}]")
    for n1, n2, a1, a2, c in ex_list:
        try:
            print(f"  Country: {c}")
            print(f"    S1 Name:    {n1}")
            print(f"    Match Name: {n2}")
            print(f"    S1 Addr:    {a1}")
            print(f"    Match Addr: {a2}")
            print("    " + "-"*40)
        except Exception:
            pass
