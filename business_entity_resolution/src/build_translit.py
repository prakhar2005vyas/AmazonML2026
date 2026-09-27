"""Learn a native-script -> Latin token dictionary from aligned training pairs (names and address components)."""
import collections
import json
import re
import unicodedata

import polars as pl
from indic_transliteration import sanscript
from rapidfuzz import fuzz, process

from config import WORK
from normalize import _script_of

NON_LATIN = re.compile(r"[ऀ-෿]")
# \w would split Indic words at combining vowel signs, so split on explicit separators instead.
SPLIT = re.compile(r"[\s,.;:()\[\]{}\-/&'\"#@!?+*|<>_=~`$%^]+")
FUZZY_ALIGN = 60  # plan v2 T1: min ratio between a romanized native token and the S1 token it is aligned to


def tokens(s):
    return [t for t in SPLIT.split(unicodedata.normalize("NFKC", s).lower()) if t]


def roman(tok):
    """Rough ITRANS romanization, letters only; used only to find which S1 token a native token lines up with."""
    scheme = _script_of(tok)
    if scheme is None:
        return tok
    return re.sub(r"[^a-z]", "", sanscript.transliterate(tok, scheme, sanscript.ITRANS).lower())


def main():
    s1 = pl.read_parquet(WORK / "train_s1.parquet")
    other = pl.concat([pl.read_parquet(WORK / f"train_s{i}.parquet") for i in (2, 3)])
    other = other.filter(pl.col("business_name").str.contains(r"[ऀ-෿]")
                         | pl.col("business_address").str.contains(r"[ऀ-෿]"))
    pairs = pl.read_parquet(WORK / "train_pairs.parquet")
    j = (pairs.join(other, left_on="other", right_on="entity_id")
              .join(s1, left_on="s1", right_on="entity_id", suffix="_s1"))
    print("pairs with native script:", len(j))

    co = collections.defaultdict(collections.Counter)
    key_n = collections.Counter()
    n_pos = n_fuzzy = 0
    for name_o, name_s1, addr_o, addr_s1 in j.select(
            "business_name", "business_name_s1", "business_address", "business_address_s1").iter_rows():
        if NON_LATIN.search(name_o):
            to, ts = tokens(name_o), tokens(name_s1)
            if len(to) == len(ts):
                n_pos += 1
                for a, b in zip(to, ts):
                    if NON_LATIN.search(a):
                        co[a][b] += 1
            elif ts:
                # plan v2 T1: different token counts (extra "india", "services", a dropped legal word...):
                # align each native token to the S1 token closest to its rough romanization
                n_fuzzy += 1
                for a in to:
                    if NON_LATIN.search(a):
                        m = process.extractOne(roman(a), ts, scorer=fuzz.ratio, score_cutoff=FUZZY_ALIGN)
                        if m:
                            co[a][m[0]] += 1
        if NON_LATIN.search(addr_o):
            s1_comps = [tokens(c) for c in addr_s1.split(",")]
            s1_comps = {" ".join(c) for c in s1_comps if c}
            for comp in addr_o.split(","):
                if NON_LATIN.search(comp):
                    key = " ".join(tokens(comp))
                    key_n[key] += 1
                    for c in s1_comps:
                        co["@" + key][c] += 1

    name_dict, addr_dict = {}, {}
    for k, cnt in co.items():
        best, n = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if k.startswith("@"):
            if n >= 5 and n / key_n[k[1:]] >= 0.6:
                addr_dict[k[1:]] = best
        elif n >= 2 and n / tot >= 0.5:
            name_dict[k] = best
    print("name pairs aligned by position", n_pos, "by fuzzy romanization", n_fuzzy)
    print("name tokens:", len(name_dict), "address components:", len(addr_dict))
    with open(WORK / "translit.json", "w", encoding="utf-8") as f:
        json.dump({"name": name_dict, "addr": addr_dict}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
