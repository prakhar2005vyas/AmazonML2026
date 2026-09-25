import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

train_s1 = r"student_resource/dataset/train/train_source1.tsv"
test_s1 = r"student_resource/dataset/test/test_source1.tsv"

def show_samples(file_path, country_filter, count=10):
    print(f"\n--- SAMPLES FOR {country_filter} from {os.path.basename(file_path)} ---")
    shown = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[3].strip() == country_filter:
                print(f"ID: {parts[0]} | Name: {parts[1]} | Addr: {parts[2]}")
                shown += 1
                if shown >= count:
                    break

show_samples(train_s1, "India", 10)
show_samples(train_s1, "US", 10)
show_samples(test_s1, "France", 10)
