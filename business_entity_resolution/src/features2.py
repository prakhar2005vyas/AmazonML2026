"""Distractor-aware pair features, computed only for pairs that survive the stage-1 prune.

The synthetic distractors differ from their S1 twin in one meaningful way: a house number shifted, one real
word swapped, or the legal form flipped. True matches instead carry typos, abbreviations, reorderings and
dropped tokens. These features separate the two: tokens are aligned fuzzily, and whatever stays unaligned is
described by how common a word it is and how close any conflicting number is.
"""
import math
import time
from multiprocessing import Pool

import jellyfish
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from config import WORK
from features import build_idf

COLS = ["name_full", "core", "compact", "addr", "nums"]
LEGAL_FLAGS = ["pvt", "ltd", "llp", "inc", "llc", "corp", "co", "lp", "pc", "public", "limited_only"]
FUZZY = 80.0
FEATURES_X = (
    ["nx_exact1", "nx_exact2", "nx_fuzzy1", "nx_fuzzy2", "nx_un1", "nx_un2", "nx_un_min_idf", "nx_un_max_idf",
     "nx_un_maxlen", "nx_un_best", "nx_compact_eq", "nx_compact_sub", "nx_order_eq"]
    + [f"lg1_{f}" for f in LEGAL_FLAGS] + [f"lg2_{f}" for f in LEGAL_FLAGS] + ["lg_symdiff"]
    + ["ux_n2_in_1", "ux_n1_in_2", "ux_n_un2", "ux_n_un1", "ux_n_near", "ux_n_mindiff", "ux_n_digit_sim",
       "ax_un1", "ax_un2", "ax_un_min_idf", "ax_exact2", "ax_len_ratio"]
)
_IDF = None


def _init(idf):
    global _IDF
    _IDF = idf


def _align(a, b):
    """For each token of a: 2 exact in b, 1 fuzzy in b, 0 unmatched; plus best fuzzy score of unmatched ones."""
    sb = set(b)
    out, best_un = [], 0.0
    for t in a:
        if t in sb:
            out.append(2)
            continue
        m = process.extractOne(t, b, scorer=fuzz.ratio) if b else None
        s = m[1] if m else 0.0
        if s >= FUZZY:
            out.append(1)
        else:
            out.append(0)
            best_un = max(best_un, s)
    return out, best_un


def _legal(tokens):
    s = set(tokens)
    f = {k: float(k in s) for k in LEGAL_FLAGS}
    f["limited_only"] = float("ltd" in s and "pvt" not in s)
    return [f[k] for k in LEGAL_FLAGS]


def _nums_feats(n1, n2):
    s1, s2 = set(n1), set(n2)
    if not s1 or not s2:
        return [-1.0, -1.0, float(len(s2)), float(len(s1)), -1.0, -1.0, -1.0]
    un2 = [x for x in s2 if x not in s1]
    un1 = [x for x in s1 if x not in s2]
    near, mindiff, dsim = 0.0, -1.0, -1.0
    if un2 and un1:
        diffs = []
        for x in un2:
            for y in un1:
                try:
                    diffs.append(abs(int(x[:12]) - int(y[:12])))
                except ValueError:
                    pass
                dsim = max(dsim, fuzz.ratio(x, y) / 100)
        if diffs:
            md = min(diffs)
            mindiff = float(np.log1p(md))
            near = float(md <= 20)
    return [len(s1 & s2) / len(s2), len(s1 & s2) / len(s1), float(len(un2)), float(len(un1)), near, mindiff, dsim]


def _pair(r):
    full1, core1, comp1, addr1, nums1, full2, core2, comp2, addr2, nums2 = r
    t1, t2 = core1.split(), core2.split()
    a1, best1 = _align(t1, t2)
    b2, best2 = _align(t2, t1)
    un = [t for t, k in zip(t1, a1) if k == 0] + [t for t, k in zip(t2, b2) if k == 0]
    idfs = [_IDF.get(t, 12.0) for t in un]
    n1c, n2c = max(len(t1), 1), max(len(t2), 1)
    name = [
        a1.count(2) / n1c, b2.count(2) / n2c, a1.count(1) / n1c, b2.count(1) / n2c,
        float(a1.count(0)), float(b2.count(0)),
        min(idfs) if idfs else -1.0, max(idfs) if idfs else -1.0,
        float(max((len(t) for t in un), default=0)), max(best1, best2) if un else -1.0,
        float(comp1 == comp2 and comp1 != ""),
        float(bool(comp1) and bool(comp2) and (comp1 in comp2 or comp2 in comp1)),
        float(t1 == t2),
    ]
    l1, l2 = _legal(full1.split()), _legal(full2.split())
    lg = l1 + l2 + [float(sum(x != y for x, y in zip(l1, l2)))]
    nums = _nums_feats(nums1.split(), nums2.split())
    w1, w2 = addr1.split(), addr2.split()
    if w1 and w2:
        c1, _ = _align(w1, w2)
        c2, _ = _align(w2, w1)
        aun = [t for t, k in zip(w1, c1) if k == 0] + [t for t, k in zip(w2, c2) if k == 0]
        addr = [float(c1.count(0)), float(c2.count(0)), min((_IDF.get(t, 12.0) for t in aun), default=-1.0),
                c2.count(2) / len(w2), min(len(w1), len(w2)) / max(len(w1), len(w2))]
    else:
        addr = [-1.0, -1.0, -1.0, -1.0, 0.0]
    return name + lg + nums + addr


def _work(rows):
    # float32 array rather than lists of Python floats: ~8x less to pickle back and to hold in the parent
    return np.asarray([_pair(r) for r in rows], dtype=np.float32).reshape(-1, len(FEATURES_X))


# --- group "x2" (plan v2, stage 1): SoftTFIDF, Monge-Elkan, homoglyph folding, phonetic keys, address shingles ---
FEATURES_X2 = ["n2_soft", "a2_soft", "n2_me12", "n2_me21", "n2_homo_eq", "n2_homo_ratio",
               "n2_phon_eq", "n2_phon_jw", "n2_phon_jacc", "a2_sh3_jacc"]
SOFT_THETA = 0.9  # Cohen et al. 2003: tokens count as matching above this Jaro-Winkler similarity
HOMO = str.maketrans({"1": "l", "i": "l", "0": "o"})  # l/I/1 and 0/O confusions ("lmpoexpo", "Lnfrastructure")
JW = JaroWinkler.normalized_similarity


def _unit_idf(toks):
    w = {t: _IDF.get(t, 12.0) for t in set(toks)}
    norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
    return {t: v / norm for t, v in w.items()}


def _soft_dir(wa, wb, tb):
    s = 0.0
    for t, v in wa.items():
        if t in wb:
            s += v * wb[t]
            continue
        m = process.extractOne(t, tb, scorer=JW, score_cutoff=SOFT_THETA)
        if m:
            s += v * wb[m[0]] * m[1]
    return s


def _soft_tfidf(a, b):
    """SoftTFIDF, averaged over both directions so the feature is symmetric. -1 when a side is empty."""
    if not a or not b:
        return -1.0
    wa, wb = _unit_idf(a), _unit_idf(b)
    return min(1.0, 0.5 * (_soft_dir(wa, wb, list(wb)) + _soft_dir(wb, wa, list(wa))))


def _monge_elkan(a, b):
    if not a or not b:
        return -1.0
    return sum(process.extractOne(t, b, scorer=JW)[1] for t in a) / len(a)


def _phon(tokens):
    return [jellyfish.metaphone(t) for t in tokens if t.isalpha()]


def _shingles(s, k=3):
    s = f" {s} "
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def _pair2(r):
    _f1, core1, comp1, addr1, _n1, _f2, core2, comp2, addr2, _n2 = r
    t1, t2 = core1.split(), core2.split()
    h1, h2 = comp1.translate(HOMO), comp2.translate(HOMO)
    p1, p2 = _phon(t1), _phon(t2)
    j1, j2 = " ".join(p1), " ".join(p2)
    sp1, sp2 = set(p1), set(p2)
    a1, a2 = addr1.split(), addr2.split()
    if addr1 and addr2:
        g1, g2 = _shingles(addr1), _shingles(addr2)
        sh = len(g1 & g2) / len(g1 | g2)
    else:
        sh = -1.0
    return [
        _soft_tfidf(t1, t2), _soft_tfidf(a1, a2), _monge_elkan(t1, t2), _monge_elkan(t2, t1),
        float(bool(h1) and h1 == h2), fuzz.ratio(h1, h2) / 100 if h1 and h2 else -1.0,
        float(bool(j1) and j1 == j2), JW(j1, j2) if j1 and j2 else -1.0,
        len(sp1 & sp2) / len(sp1 | sp2) if sp1 and sp2 else -1.0, sh,
    ]


def _work2(rows):
    return np.asarray([_pair2(r) for r in rows], dtype=np.float32).reshape(-1, len(FEATURES_X2))


GROUPS = {"x": (_work, FEATURES_X), "x2": (_work2, FEATURES_X2)}


def iter_extra(pairs, split, workers=6, batch_size=500_000, skip=(), group="x"):
    """Yields (start, frame of the group's features) for consecutive row slices of pairs, in order.

    Batching keeps the parent's string rows and results bounded to one slice at a time (the full test set
    OOM'd when materialised at once). Starts listed in skip are not computed (resume support).
    """
    work, names = GROUPS[group]
    t = time.time()
    s1_norm = pl.read_parquet(WORK / f"{split}_s1_norm.parquet", columns=["entity_id"] + COLS)
    other_norm = pl.concat([pl.read_parquet(WORK / f"{split}_s{i}_norm.parquet", columns=["entity_id"] + COLS)
                            for i in (2, 3)])
    idf = build_idf(split)
    n = len(pairs)
    with Pool(workers, initializer=_init, initargs=(idf,)) as pool:
        for start in range(0, n, batch_size):
            if start in skip:
                continue
            j = (pairs.slice(start, batch_size).select("s1", "other")
                 .join(s1_norm, left_on="s1", right_on="entity_id", how="left", maintain_order="left")
                 .join(other_norm, left_on="other", right_on="entity_id", how="left", suffix="_o", maintain_order="left"))
            rows = j.select(COLS + [c + "_o" for c in COLS]).fill_null("").rows()
            del j
            chunks = [rows[k:k + 20000] for k in range(0, len(rows), 20000)]
            del rows
            x = np.concatenate(list(pool.imap(work, chunks)))
            del chunks
            print(f"  extra_features[{group}] {min(start + batch_size, n)}/{n} {time.time() - t:.0f}s", flush=True)
            yield start, pl.DataFrame(x, schema=names)


def extra_features(pairs, split, workers=6, batch_size=500_000, group="x"):
    """pairs: frame with s1, other. Returns pairs with the group's features appended (same row order)."""
    t = time.time()
    x = pl.concat([f for _, f in iter_extra(pairs, split, workers, batch_size, group=group)])
    print(f"extra features[{group}]", len(pairs), f"{time.time() - t:.0f}s", flush=True)
    return pl.concat([pairs, x], how="horizontal")
