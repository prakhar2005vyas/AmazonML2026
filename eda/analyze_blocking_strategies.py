import sys
import os
import re
import time
from collections import Counter, defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

train_dir = r"student_resource/dataset/train"

print("=== BLOCKING & CANDIDATE GENERATION FEASIBILITY STUDY ===")

# Generic stop words and business entity terms to clean
STOPWORDS = {
    # English & general
    "the", "and", "of", "in", "for", "at", "by", "on", "with", "to", "a", "an",
    # India legal / common
    "pvt", "ltd", "private", "limited", "enterprises", "solutions", "services", 
    "traders", "trading", "industries", "company", "co", "llp", "india",
    # US legal / common
    "inc", "corp", "corporation", "llc", "group", "holdings", "associates",
    "partners", "management", "consulting", "services", "enterprises",
    # French legal / common
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "france", "societe"
}

def clean_tokens(text, min_len=3):
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return {t for t in tokens if len(t) >= min_len and t not in STOPWORDS}

def get_char_ngrams(text, n=3):
    clean = re.sub(r'[^a-z0-9]', '', text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

# Load a sample of 25,000 true match pairs
print("Sampling 25,000 ground truth pairs...")
sampled_pairs = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i % 88 == 0:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1].strip():
                s1_id = parts[0]
                matches = [m.strip() for m in parts[1].split(",") if m.strip()]
                for m in matches:
                    sampled_pairs.append((s1_id, m))
            if len(sampled_pairs) >= 25000:
                break

print(f"Sampled {len(sampled_pairs):,d} true match pairs.")

needed_s1 = {s1 for s1, m in sampled_pairs}
needed_s2 = {m for s1, m in sampled_pairs if m.startswith("S2-")}
needed_s3 = {m for s1, m in sampled_pairs if m.startswith("S3-")}

def load_records(file_path, id_set):
    records = {}
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in id_set:
                records[parts[0]] = {
                    "name": parts[1] if len(parts) > 1 else "",
                    "addr": parts[2] if len(parts) > 2 else "",
                    "country": parts[3] if len(parts) > 3 else ""
                }
    return records

s1_data = load_records(os.path.join(train_dir, "train_source1.tsv"), needed_s1)
s2_data = load_records(os.path.join(train_dir, "train_source2.tsv"), needed_s2)
s3_data = load_records(os.path.join(train_dir, "train_source3.tsv"), needed_s3)

# Test blocking recall of various keys on the sampled true pairs
n_pairs = len(sampled_pairs)

shared_name_tokens_3 = 0
shared_name_tokens_2 = 0
shared_name_ngrams_3 = 0
shared_name_ngrams_4 = 0
shared_addr_tokens_3 = 0
shared_name_or_addr_token = 0
exact_name_prefix_3 = 0
exact_name_prefix_4 = 0

dba_cases = []

for s1_id, match_id in sampled_pairs:
    r1 = s1_data.get(s1_id)
    r2 = s2_data.get(match_id) if match_id.startswith("S2-") else s3_data.get(match_id)
    if not r1 or not r2:
        continue
    
    n1, n2 = r1["name"], r2["name"]
    a1, a2 = r1["addr"], r2["addr"]
    
    t1_name_3 = clean_tokens(n1, 3)
    t2_name_3 = clean_tokens(n2, 3)
    has_shared_name_3 = bool(t1_name_3 & t2_name_3)
    if has_shared_name_3:
        shared_name_tokens_3 += 1
        
    t1_name_2 = clean_tokens(n1, 2)
    t2_name_2 = clean_tokens(n2, 2)
    has_shared_name_2 = bool(t1_name_2 & t2_name_2)
    if has_shared_name_2:
        shared_name_tokens_2 += 1
        
    ng1_3 = get_char_ngrams(n1, 3)
    ng2_3 = get_char_ngrams(n2, 3)
    has_shared_ng_3 = bool(ng1_3 & ng2_3)
    if has_shared_ng_3:
        shared_name_ngrams_3 += 1
        
    ng1_4 = get_char_ngrams(n1, 4)
    ng2_4 = get_char_ngrams(n2, 4)
    if bool(ng1_4 & ng2_4):
        shared_name_ngrams_4 += 1
        
    t1_addr_3 = clean_tokens(a1, 3)
    t2_addr_3 = clean_tokens(a2, 3)
    has_shared_addr_3 = bool(t1_addr_3 & t2_addr_3)
    if has_shared_addr_3:
        shared_addr_tokens_3 += 1
        
    if has_shared_name_3 or has_shared_addr_3:
        shared_name_or_addr_token += 1
        
    # Clean prefix
    c1 = re.sub(r'[^a-z0-9]', '', n1.lower())
    c2 = re.sub(r'[^a-z0-9]', '', n2.lower())
    if c1[:3] and c1[:3] == c2[:3]:
        exact_name_prefix_3 += 1
    if c1[:4] and c1[:4] == c2[:4]:
        exact_name_prefix_4 += 1
        
    # Catch DBA cases where name has NO token overlap at all
    if not has_shared_name_3 and has_shared_addr_3 and len(dba_cases) < 5:
        dba_cases.append((n1, n2, a1, a2, r1["country"]))

print("\n--- RECALL CEILING OF CANDIDATE GENERATION STRATEGIES ---")
print(f"Total True Pairs Evaluated: {n_pairs:,d}")
print(f"1. Share >=1 Name Token (len >= 3, no stopwords): {shared_name_tokens_3:6,d} ({shared_name_tokens_3/n_pairs*100:6.2f}%)")
print(f"2. Share >=1 Name Token (len >= 2, no stopwords): {shared_name_tokens_2:6,d} ({shared_name_tokens_2/n_pairs*100:6.2f}%)")
print(f"3. Share >=1 Name Character 3-Gram:              {shared_name_ngrams_3:6,d} ({shared_name_ngrams_3/n_pairs*100:6.2f}%)")
print(f"4. Share >=1 Name Character 4-Gram:              {shared_name_ngrams_4:6,d} ({shared_name_ngrams_4/n_pairs*100:6.2f}%)")
print(f"5. Exact Name Prefix (first 3 chars):             {exact_name_prefix_3:6,d} ({exact_name_prefix_3/n_pairs*100:6.2f}%)")
print(f"6. Exact Name Prefix (first 4 chars):             {exact_name_prefix_4:6,d} ({exact_name_prefix_4/n_pairs*100:6.2f}%)")
print(f"7. Share >=1 Address Token (len >= 3):            {shared_addr_tokens_3:6,d} ({shared_addr_tokens_3/n_pairs*100:6.2f}%)")
print(f"8. UNION: (Share Name Token) OR (Share Addr Token): {shared_name_or_addr_token:6,d} ({shared_name_or_addr_token/n_pairs*100:6.2f}%)")

print("\n--- EXAMPLES OF TRUE MATCHES WITH ZERO NAME TOKEN OVERLAP (DBA / TRADE NAMES) ---")
for n1, n2, a1, a2, c in dba_cases:
    print(f"Country: {c}")
    print(f"  S1:   Name='{n1}' | Addr='{a1}'")
    print(f"  Match:Name='{n2}' | Addr='{a2}'")
    print("-" * 50)
