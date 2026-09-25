import os
import time
from collections import Counter

gt_path = r"student_resource/dataset/train/train_ground_truth.tsv"
print("Analyzing train_ground_truth.tsv...")
t0 = time.time()

num_s1 = 0
match_counts = Counter()
s2_count = 0
s3_count = 0
combo_counts = Counter() # '0', 'S2_only', 'S3_only', 'both'

s2_matched_set = set()
s3_matched_set = set()
s2_multimatch = Counter()
s3_multimatch = Counter()

with open(gt_path, "r", encoding="utf-8") as f:
    header = f.readline().strip()
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        s1_id = parts[0]
        num_s1 += 1
        
        if len(parts) < 2 or not parts[1].strip():
            # Singleton (0 matches)
            match_counts[0] += 1
            combo_counts['none (singleton)'] += 1
            continue
        
        matches = [m.strip() for m in parts[1].split(",") if m.strip()]
        k = len(matches)
        match_counts[k] += 1
        
        has_s2 = False
        has_s3 = False
        for m in matches:
            if m.startswith("S2-"):
                s2_count += 1
                has_s2 = True
                s2_matched_set.add(m)
                s2_multimatch[m] += 1
            elif m.startswith("S3-"):
                s3_count += 1
                has_s3 = True
                s3_matched_set.add(m)
                s3_multimatch[m] += 1
            else:
                print(f"Unknown match ID prefix: {m}")
        
        if has_s2 and has_s3:
            combo_counts['both S2 and S3'] += 1
        elif has_s2:
            combo_counts['S2 only'] += 1
        elif has_s3:
            combo_counts['S3 only'] += 1

t1 = time.time()

print(f"\n--- Ground Truth Summary (processed in {t1-t0:.2f}s) ---")
print(f"Total S1 entities in Ground Truth: {num_s1:,d}")
singletons = match_counts[0]
matched_s1 = num_s1 - singletons
print(f"Singletons (0 matches): {singletons:,d} ({singletons / num_s1 * 100:.2f}%)")
print(f"S1 with >= 1 match:    {matched_s1:,d} ({matched_s1 / num_s1 * 100:.2f}%)")

print("\n--- Distribution of Match Counts per S1 entity ---")
for k in sorted(match_counts.keys()):
    if k <= 10 or match_counts[k] > 100:
        pct = match_counts[k] / num_s1 * 100
        print(f"  {k} matches: {match_counts[k]:10,d} ({pct:6.2f}%)")

print(f"\nTotal match links: {s2_count + s3_count:,d}")
print(f"  Matches to S2:   {s2_count:,d}")
print(f"  Matches to S3:   {s3_count:,d}")
print(f"  Unique S2 matched: {len(s2_matched_set):,d}")
print(f"  Unique S3 matched: {len(s3_matched_set):,d}")

print("\n--- Match Source Combinations per S1 ---")
for combo, cnt in combo_counts.items():
    print(f"  {combo:<20}: {cnt:10,d} ({cnt / num_s1 * 100:6.2f}%)")

# Check multi-match (is any S2 or S3 entity matched to MORE THAN ONE S1 entity?)
s2_multi = sum(1 for m, c in s2_multimatch.items() if c > 1)
s3_multi = sum(1 for m, c in s3_multimatch.items() if c > 1)
print(f"\nS2 entities matched to >1 S1 entity: {s2_multi} (max: {max(s2_multimatch.values()) if s2_multimatch else 0})")
print(f"S3 entities matched to >1 S1 entity: {s3_multi} (max: {max(s3_multimatch.values()) if s3_multimatch else 0})")

# Save summary to file for downstream reporting
with open("eda/gt_summary.txt", "w") as out:
    out.write(f"num_s1={num_s1}\n")
    out.write(f"singletons={singletons}\n")
    out.write(f"matched_s1={matched_s1}\n")
    out.write(f"s2_count={s2_count}\n")
    out.write(f"s3_count={s3_count}\n")
    out.write(f"unique_s2_matched={len(s2_matched_set)}\n")
    out.write(f"unique_s3_matched={len(s3_matched_set)}\n")
