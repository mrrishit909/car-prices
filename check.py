"""Step 5: the project's check. Recompute with different code, in plain Python straight from the downloaded file (csv, gzip, hashlib, math and loops: no pandas, numpy, scipy or sklearn),
and compare with results/:

    python check.py          (needs data/cars_slim.csv.gz from download.py and data/oof_*.csv.gz from analyze.py; takes a few minutes)

recounted from the raw file: every cleaning count and the re-listing counts (a union-find over VIN and make-model-year-odometer keys); the cars, the 250 model keys, the md5 folds and the fold sizes;
the ridge fits M1, M2 and M3 (X'X + lambda D by Gaussian elimination / Cholesky, built from the per-fold sums) and the mean model M0 under designs D3, D4 and FWD, with every metric;
the metrics of the gradient-boosting model M4 for all five designs, recomputed from the stored out-of-fold predictions (the fit itself is sklearn's HistGradientBoostingRegressor and is not
re-derived: it is verified by the fact that the stored rows are exactly the cars and folds recomputed here, never trained on their own fold, and that every number built from them agrees);
the permutation-importance table; the split-conformal intervals, their coverage and the bands; and the depreciation tables, rates and single-feature R-squared.
Comparisons are at a relative tolerance of 1e-8 (counts exact).
"""
import csv
import gzip
import hashlib
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
K, LAM, TOPK, CUT, REFYEAR = 5, 100.0, 250, "2021-04-26", 2021
VIN_RE = re.compile(r"[A-HJ-NPR-Z0-9]{17}")
CATS = ["cylinders", "fuel", "transmission", "drive", "type", "condition", "state"]
n_ok = 0


def same(label, got, want, rel=1e-8):
    global n_ok
    n_ok += 1
    if isinstance(want, str) and not re.fullmatch(r"-?[0-9.eE+-]+", want):
        assert str(got) == want, f"{label}: recount {got} != results {want}"
        return
    want = float(want)
    assert abs(float(got) - want) <= rel * abs(want) + 1e-12, f"{label}: recount {got} != results {want}"


def table(name):
    return list(csv.DictReader(open(R / f"{name}.csv")))


def norm(s):
    return " ".join(s.lower().split())


def cat(s):
    n = norm(s) if s else ""
    return n or None


def median(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def fold_of(key):
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16) % K


def conf_q(absres):
    s = sorted(absres); n = len(s)
    return s[((n + 1) * 9 + 9) // 10 - 1]


def metrics(ly, p):
    n = len(ly); e = [pi - yi for pi, yi in zip(p, ly)]; my = sum(ly) / n; sst = sum((y - my) ** 2 for y in ly); sse = sum(x * x for x in e)
    price = [math.exp(y) for y in ly]; pred = [math.exp(x) for x in p]; ad = [abs(a - b) for a, b in zip(pred, price)]
    return {"n": n, "rmse_log": math.sqrt(sse / n), "r2_log": 1 - sse / sst, "mae_usd": sum(ad) / n, "mdape": median([a / b for a, b in zip(ad, price)]),
            "within10": sum(abs(a / b - 1) <= 0.10 for a, b in zip(pred, price)) / n, "within20": sum(abs(a / b - 1) <= 0.20 for a, b in zip(pred, price)) / n}


def solve_spd(A, b):
    """Cholesky A = L L' then two triangular solves."""
    n = len(b); L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            Lj = L[j]; s = sum(Li[k] * Lj[k] for k in range(j))
            if i == j: Li[i] = math.sqrt(A[i][i] - s)
            else: Li[j] = (A[i][j] - s) / Lj[j]
    z = [0.0] * n
    for i in range(n): z[i] = (b[i] - sum(L[i][k] * z[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1): x[i] = (z[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def ols(X, y):
    """plain least squares with an intercept by the normal equations; returns coefficients and R-squared."""
    n, k = len(y), len(X[0]) + 1; A = [[0.0] * k for _ in range(k)]; b = [0.0] * k
    for xr, yi in zip(X, y):
        v = [1.0] + list(xr)
        for i in range(k):
            b[i] += v[i] * yi
            for j in range(k): A[i][j] += v[i] * v[j]
    c = solve_spd(A, b); my = sum(y) / n
    sse = sum((yi - sum(ci * vi for ci, vi in zip(c, [1.0] + list(xr)))) ** 2 for xr, yi in zip(X, y)); sst = sum((yi - my) ** 2 for yi in y)
    return c, 1 - sse / sst


def main():
    # ---------------- read and clean
    raw = list(csv.DictReader(gzip.open(D / "cars_slim.csv.gz", "rt", newline="")))
    names = ["price is 0", "price 1 to 499", "price above 150,000", "placeholder price (1234, 12345, 11111...)", "model year missing or before 1995", "odometer missing", "odometer above 400,000",
             "odometer under 100 on a 2019 or older car", "make or model missing", "title not clean (salvage, rebuilt, lien, missing...)"]
    drops, left, cl = [0] * 10, len(raw), []
    for r in raw:
        price = int(float(r["price"])); year = float(r["year"]) if r["year"] else None; odo = float(r["odometer"]) if r["odometer"] else None
        make, model, title = cat(r["manufacturer"]), cat(r["model"]), cat(r["title_status"]); t = str(price)
        hits = [price == 0, 0 < price < 500, price > 150000, price in (1234, 12345, 123456, 1234567) or (len(t) >= 5 and len(set(t)) == 1), year is None or year < 1995, odo is None, odo is not None and odo > 400000,
                odo is not None and year is not None and odo < 100 and year <= 2019, make is None or model is None, title is None or title != "clean"]
        for i, h in enumerate(hits):
            if h: drops[i] += 1; break
        else:
            row = {"id": int(r["id"]), "price": price, "year": int(year), "odo": odo, "make": make, "mstr": model, "VIN": r["VIN"], "day": r["posting_date"][:10] or None}
            for c in CATS: row[c] = cat(r[c]) or "unk"
            row["age"] = max(0, REFYEAR - row["year"]); cl.append(row)
    audit = table("audit"); same("raw", len(raw), audit[0]["left"]); n = len(raw)
    for i, nm in enumerate(names):
        n -= drops[i]; assert audit[i + 1]["step"] == nm; same("drop " + nm, drops[i], audit[i + 1]["dropped"]); same("left " + nm, n, audit[i + 1]["left"])
    assert len({r["id"] for r in cl}) == len(cl)
    # ---------------- re-listings: union-find over VIN and near keys
    N = len(cl); parent = list(range(N))

    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)

    vin = [None] * N
    for i, r in enumerate(cl):
        s = r["VIN"].strip().upper() if r["VIN"] else ""
        if s and VIN_RE.fullmatch(s) and len(set(s)) > 1: vin[i] = s
    my = {}
    for i, r in enumerate(cl):
        if vin[i]: my.setdefault(vin[i], set()).add((r["make"], r["year"]))
    bad = {v for v, s in my.items() if len(s) > 1}
    vin_key = [v if v and v not in bad else None for v in vin]

    def pattern_odo(o):
        t = str(int(o)); return o % 1000 == 0 or len(set(t)) == 1 or t in ("1234", "12345", "123456", "1234567")

    near = [None if pattern_odo(r["odo"]) else f'{r["make"]}|{r["mstr"]}|{r["year"]}|{int(r["odo"])}' for r in cl]
    for keys in (vin_key, near):
        first = {}
        for i, k in enumerate(keys):
            if k is None: continue
            if k in first: union(i, first[k])
            else: first[k] = i
    comp = [find(i) for i in range(N)]; csize = {}
    for c in comp: csize[c] = csize.get(c, 0) + 1
    minid = {}
    for i, c in enumerate(comp): minid[c] = min(minid.get(c, 10 ** 18), cl[i]["id"])
    firstday = {}
    for i, c in enumerate(comp):
        d = cl[i]["day"]
        if d is not None and (c not in firstday or d < firstday[c]): firstday[c] = d
    for i, r in enumerate(cl): r["car"] = minid[comp[i]]; r["first_date"] = firstday.get(comp[i])
    cars = [r for r in cl if r["id"] == r["car"]]
    vk = {(vin_key[i] if vin_key[i] else f"#{i}") for i in range(N)}
    vincnt = {}
    for v in vin:
        if v: vincnt[v] = vincnt.get(v, 0) + 1
    S = {r["key"]: r["value"] for r in table("summary")}
    for k, v in (("raw", len(raw)), ("clean_listings", N), ("cars", len(cars)), ("vin_valid_listings", sum(v is not None for v in vin)), ("vin_distinct_valid", len(vincnt)), ("vin_unreliable", len(bad)),
                 ("vin_unreliable_listings", sum(1 for v in vin if v in bad)), ("near_key_listings", sum(k is not None for k in near)), ("largest_car_listings", max(csize.values())),
                 ("listings_in_multi", sum(1 for c in comp if csize[c] > 1)), ("cars_listed_more_than_once", sum(1 for s in csize.values() if s > 1)), ("top_vin_listings", max(vincnt.values())),
                 ("cars_by_vin_rule_only", len(vk))):
        same("summary " + k, v, S[k])
    # ---------------- features
    cnt = {}
    for r in cars: cnt[(r["make"], r["mstr"])] = cnt.get((r["make"], r["mstr"]), 0) + 1
    top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:TOPK]; keys = sorted(f"{a}|{b}" for (a, b), _ in top); keyset = set(keys)
    for r in cl: r["model_key"] = f'{r["make"]}|{r["mstr"]}' if f'{r["make"]}|{r["mstr"]}' in keyset else "other"
    lev = {c: sorted({r[c] for r in cl}) for c in CATS + ["make"]}
    same("model_keys", len(keys), S["model_keys"]); same("makes", len(lev["make"]), S["makes"]); same("model_keys_share_of_cars", sum(r["model_key"] != "other" for r in cars) / len(cars), S["model_keys_share_of_cars"])
    same("median_price", median([r["price"] for r in cars]), S["median_price"]); same("mean_price", sum(r["price"] for r in cars) / len(cars), S["mean_price"])
    same("median_age", median([r["age"] for r in cars]), S["median_age"]); same("median_odo", median([r["odo"] for r in cars]), S["median_odo"])
    ma, mo = sum(r["age"] for r in cars) / len(cars), sum(r["odo"] for r in cars) / len(cars)
    sab = sum((r["age"] - ma) * (r["odo"] - mo) for r in cars); saa = sum((r["age"] - ma) ** 2 for r in cars); sbb = sum((r["odo"] - mo) ** 2 for r in cars)
    same("corr_age_odo", sab / math.sqrt(saa * sbb), S["corr_age_odo"])
    assert S["first_post"] == min(r["day"] for r in cl if r["day"]) and S["last_post"] == max(r["day"] for r in cl if r["day"]); same("lambda", LAM, S["lambda"])
    # twins: share of clean listings whose car has a listing in another random-listing fold
    cf = {}
    for r in cl: cf.setdefault(r["car"], set()).add(fold_of(r["id"]))
    same("d1_test_listings_with_twin_in_train", sum(len(cf[r["car"]]) > 1 for r in cl) / N, S["d1_test_listings_with_twin_in_train"])
    # ---------------- folds
    designs = {"D1": (cl, lambda r: r["id"]), "D2": (cl, lambda r: r["car"]), "D3": (cars, lambda r: r["car"]), "D4": (cars, lambda r: f'{r["make"]}|{r["mstr"]}')}
    fld = {}
    for dn, (rows, kf) in designs.items():
        fld[dn] = [fold_of(kf(r)) for r in rows]; seen = {}
        for r, f in zip(rows, fld[dn]):
            g = kf(r); assert seen.setdefault(g, f) == f, f"{dn}: group {g} in two folds"
    for fr in table("folds"):
        rows, kf = designs[fr["design"]]; k = int(fr["fold"]); sel = [r for r, f in zip(rows, fld[fr["design"]]) if f == k]
        same(f'folds {fr["design"]} {k} rows', len(sel), fr["rows"]); same(f'folds {fr["design"]} {k} cars', len({r["car"] for r in sel}), fr["cars"])
    # ---------------- ridge ladder (nested columns: M1 = first 5, M2 = + make and engine, M3 = + model keys)
    def build(r):
        a, m = r["age"] / 10, r["odo"] / 100000; idx, val = [0, 1, 2, 3, 4], [1.0, a, a * a, m, m * m]; off = 5
        for c in ["make", "cylinders", "fuel", "transmission", "drive"]:
            levs = lev[c][1:]
            if r[c] in levs: idx.append(off + levs.index(r[c])); val.append(1.0)
            off += len(levs)
        if r["model_key"] != "other": idx.append(off + keys.index(r["model_key"])); val.append(1.0)
        return idx, val

    p1 = 5; p2 = 5 + sum(len(lev[c]) - 1 for c in ["make", "cylinders", "fuel", "transmission", "drive"]); p3 = p2 + len(keys); sp = [build(r) for r in cars]
    lylist = [math.log(r["price"]) for r in cars]; pos = {r["id"]: i for i, r in enumerate(cars)}

    def accumulate(sel):
        A = [[0.0] * p3 for _ in range(p3)]; b = [0.0] * p3
        for i in sel:
            idx, val = sp[i]; y = lylist[i]
            for a, va in zip(idx, val):
                b[a] += va * y; Aa = A[a]
                for c, vc in zip(idx, val): Aa[c] += va * vc
        return A, b

    def fit(A, b, q):
        M = [row[:q] for row in A[:q]]
        for i in range(1, q): M[i][i] += LAM
        return solve_spd(M, b[:q])

    def pred(coef, q, i):
        idx, val = sp[i]; return sum(c * coef[j] for j, c in zip(idx, val) if j < q) if False else sum(coef[j] * v for j, v in zip(idx, val) if j < q)

    def ladder(trainset, testset, tag, fits=None):
        out = {"M0 the mean": [], "M1 OLS: age, miles": [], "M2 ridge: + make, engine": [], "M3 ridge: + make-model": []}
        A, b = fits if fits else accumulate(trainset); m0 = sum(lylist[i] for i in trainset) / len(trainset)
        co = [fit(A, b, q) for q in (p1, p2, p3)]
        return m0, co

    def combine(A1, b1, A2, b2, sign):
        return [[x + sign * y for x, y in zip(r1, r2)] for r1, r2 in zip(A1, A2)], [x + sign * y for x, y in zip(b1, b2)]

    mr = {(r["design"], r["model"]): r for r in table("models")}; cols = ["n", "rmse_log", "r2_log", "mae_usd", "mdape", "within10", "within20"]

    def compare(design, model, ly, p):
        m = metrics(ly, p)
        for c in cols: same(f"{design} {model} {c}", m[c], mr[(design, model)][c], 1e-8)

    for dn, label in (("D3", "D3 by car, one listing each"), ("D4", "D4 by make-model (unseen model)")):
        fl = fld[dn]; tot = accumulate(range(len(cars))); per = []
        for k in range(K): per.append(accumulate([i for i in range(len(cars)) if fl[i] == k]))
        P = {m: [0.0] * len(cars) for m in ("M0 the mean", "M1 OLS: age, miles", "M2 ridge: + make, engine", "M3 ridge: + make-model")}
        for k in range(K):
            A, b = combine(tot[0], tot[1], per[k][0], per[k][1], -1.0); tr = [i for i in range(len(cars)) if fl[i] != k]; te = [i for i in range(len(cars)) if fl[i] == k]
            m0 = sum(lylist[i] for i in tr) / len(tr)
            for mname, q in (("M1 OLS: age, miles", p1), ("M2 ridge: + make, engine", p2), ("M3 ridge: + make-model", p3)):
                co = fit(A, b, q)
                for i in te: P[mname][i] = pred(co, q, i)
            for i in te: P["M0 the mean"][i] = m0
        for mname, v in P.items(): compare(label, mname, lylist, v)
        if dn == "D3": ridge_d3 = P
        print("check: ridge", dn, "done", flush=True)
    tr = [i for i, r in enumerate(cars) if r["first_date"] is not None and r["first_date"] <= CUT]; te = [i for i, r in enumerate(cars) if r["first_date"] is not None and r["first_date"] > CUT]
    same("fwd_train", len(tr), S["fwd_train"]); same("fwd_test", len(te), S["fwd_test"]); assert S["fwd_cut"] == CUT
    A, b = accumulate(tr); m0 = sum(lylist[i] for i in tr) / len(tr); lyt = [lylist[i] for i in te]
    compare("FWD later cars", "M0 the mean", lyt, [m0] * len(te))
    for mname, q in (("M1 OLS: age, miles", p1), ("M2 ridge: + make, engine", p2), ("M3 ridge: + make-model", p3)):
        co = fit(A, b, q); compare("FWD later cars", mname, lyt, [pred(co, q, i) for i in te])
    # ---------------- boosting: metrics from the stored out-of-fold predictions, after tying the rows to the recount
    def load_oof(name, rows, fl, kf):
        o = list(csv.DictReader(gzip.open(D / f"oof_{name}.csv.gz", "rt", newline=""))); byid = {r["id"]: (r, f) for r, f in zip(rows, fl)}
        if name == "fwd": assert {int(x["id"]) for x in o} == {cars[i]["id"] for i in te}
        else: assert len(o) == len(rows) and {int(x["id"]) for x in o} == set(byid)
        for x in o:
            r, f = byid[int(x["id"])]; assert r["price"] == float(x["price"]) and r["car"] == int(x["car"]); assert name == "fwd" or f == int(x["fold"])
            assert abs(float(x["ly"]) - math.log(r["price"])) < 1e-12 and x["m4"] != ""
        return o

    oofs = {nm: load_oof(nm, rows, fld[dn], None) for nm, dn, rows in (("d1", "D1", cl), ("d2", "D2", cl), ("d3", "D3", cars), ("d4", "D4", cars))}
    oofs["fwd"] = load_oof("fwd", cars, [0] * len(cars), None)
    for nm, label in (("d1", "D1 random listings (leaky)"), ("d2", "D2 by car, all listings"), ("d3", "D3 by car, one listing each"), ("d4", "D4 by make-model (unseen model)"), ("fwd", "FWD later cars")):
        o = oofs[nm]; compare(label, "M4 boosting", [float(x["ly"]) for x in o], [float(x["m4"]) for x in o])
    o3 = oofs["d3"]; ly3 = [float(x["ly"]) for x in o3]; base = metrics(ly3, [float(x["m4"]) for x in o3])["rmse_log"]
    for r in table("importance"):
        v = metrics(ly3, [float(x["perm: " + r["feature"]]) for x in o3])["rmse_log"]; same("imp " + r["feature"], v, r["rmse_log_shuffled"]); same("imp base", base, r["rmse_log_base"]); same("imp rise " + r["feature"], v - base, r["rise"], 1e-6)
    ranks = [round(float(r["rise"]), 6) for r in table("importance")]; assert ranks == sorted(ranks, reverse=True)
    # ---------------- intervals
    fold3 = [int(x["fold"]) for x in o3]; price3 = [float(x["price"]) for x in o3]; IR = {(r["model"], r["scope"]): r for r in table("intervals")}
    m1d3 = ridge_d3["M1 OLS: age, miles"]; m3d3 = ridge_d3["M3 ridge: + make-model"]; ids3 = [int(x["id"]) for x in o3]; m4d3 = [float(x["m4"]) for x in o3]
    order = [pos[i] for i in ids3]
    for mname, preds in (("M1 OLS: age, miles", [m1d3[j] for j in order]), ("M3 ridge: + make-model", [m3d3[j] for j in order]), ("M4 boosting", m4d3)):
        res = [y - p for y, p in zip(ly3, preds)]; qk = {k: conf_q([abs(e) for e, f in zip(res, fold3) if f != k]) for k in range(K)}; q = [qk[f] for f in fold3]
        cov = [abs(e) <= qq for e, qq in zip(res, q)]; pr = [math.exp(p) for p in preds]; lo = [a * math.exp(-qq) for a, qq in zip(pr, q)]; hi = [a * math.exp(qq) for a, qq in zip(pr, q)]; qa = conf_q([abs(e) for e in res])
        row = IR[(mname, "D3 cars, folds by car")]
        same(mname + " cov", sum(cov) / len(cov), row["coverage"]); same(mname + " q", qa, row["q_log"]); same(mname + " width", median([h - l for h, l in zip(hi, lo)]), row["median_width_usd"])
        same(mname + " relhw", median([(h - l) / 2 / a for h, l, a in zip(hi, lo, pr)]), row["median_relative_halfwidth"]); same(mname + " n", len(res), row["n"])
        if mname == "M4 boosting":
            bands = {r["predicted_price"]: r for r in table("interval_bands")}
            for lab, a, b2 in (("under $5k", 0, 5000), ("$5k-10k", 5000, 10000), ("$10k-20k", 10000, 20000), ("$20k-40k", 20000, 40000), ("$40k and over", 40000, 1e12)):
                s = [i for i in range(len(res)) if a <= pr[i] < b2]; br = bands[lab]
                same(lab + " n", len(s), br["n"]); same(lab + " cov", sum(cov[i] for i in s) / len(s), br["coverage"]); same(lab + " width", median([hi[i] - lo[i] for i in s]), br["median_width_usd"])
                same(lab + " lo", median([lo[i] for i in s]), br["median_lo"]); same(lab + " hi", median([hi[i] for i in s]), br["median_hi"])
            o1 = oofs["d1"]; ql = conf_q([abs(float(x["ly"]) - float(x["m4"])) for x in o1]); row = IR[(mname, "interval calibrated on D1 random folds, tried on D3 cars")]
            same("leaky q", ql, row["q_log"]); same("leaky cov", sum(abs(e) <= ql for e in res) / len(res), row["coverage"]); same("leaky width", median([a * (math.exp(ql) - math.exp(-ql)) for a in pr]), row["median_width_usd"])
            same("leaky relhw", (math.exp(ql) - math.exp(-ql)) / 2, row["median_relative_halfwidth"]); same("d3 q summary", qa, S["q_d3_m4"])
            o4 = oofs["d4"]; r4 = [float(x["ly"]) - float(x["m4"]) for x in o4]; p4 = [math.exp(float(x["m4"])) for x in o4]; row = IR[(mname, "interval calibrated on D3, tried on D4 unseen models")]
            same("d4 cov", sum(abs(e) <= qa for e in r4) / len(r4), row["coverage"]); same("d4 width", median([a * (math.exp(qa) - math.exp(-qa)) for a in p4]), row["median_width_usd"]); same("d4 n", len(r4), row["n"])
    # ---------------- depreciation
    mk = {}
    for r in cars: mk[r["make"]] = mk.get(r["make"], 0) + 1
    topm = sorted(mk.items(), key=lambda kv: (-kv[1], kv[0])); top8 = [m for m, _ in topm[:8]]; top12 = [m for m, _ in topm[:12]]
    da = table("dep_age"); want = []
    for m in top8:
        for a in range(0, 21):
            ps = [r["price"] for r in cars if r["make"] == m and r["age"] == a]
            if len(ps) >= 30: want.append((m, a, len(ps), median(ps)))
    assert len(da) == len(want)
    for row, (m, a, nn, md) in zip(da, want): assert row["make"] == m and int(row["age"]) == a; same(f"dep_age {m} {a} n", nn, row["n"]); same(f"dep_age {m} {a}", md, row["median_price"])
    edges = [0, 20000, 40000, 60000, 80000, 100000, 125000, 150000, 200000, 300000, 400001]; dm = table("dep_miles"); want = []
    for lab, lo_a, hi_a in (("age 3-5", 3, 5), ("age 6-9", 6, 9), ("age 10-14", 10, 14)):
        for i in range(len(edges) - 1):
            ps = [r["price"] for r in cars if lo_a <= r["age"] <= hi_a and edges[i] <= r["odo"] < edges[i + 1]]
            if len(ps) >= 30: want.append((lab, edges[i], edges[i + 1], len(ps), median(ps)))
    assert len(dm) == len(want)
    for row, (lab, a, b2, nn, md) in zip(dm, want): assert row["age_band"] == lab and int(row["odo_from"]) == a; same(f"dep_miles {lab} {a} n", nn, row["n"]); same(f"dep_miles {lab} {a}", md, row["median_price"])
    for row in table("dep_rate"):
        s = cars if row["make"] == "all makes" else [r for r in cars if r["make"] == row["make"]]
        c, r2 = ols([(r["age"], r["odo"] / 10000) for r in s], [math.log(r["price"]) for r in s])
        same("rate n " + row["make"], len(s), row["n"]); same("pct/yr " + row["make"], 100 * (math.exp(c[1]) - 1), row["pct_per_year"]); same("pct/10k " + row["make"], 100 * (math.exp(c[2]) - 1), row["pct_per_10k_miles"]); same("r2 " + row["make"], r2, row["r2_log"])
    assert [r["make"] for r in table("dep_rate")][1:] == top12
    for row in table("single_r2"):
        X = {"age only": [(r["age"],) for r in cars], "miles only": [(r["odo"] / 10000,) for r in cars], "age and miles": [(r["age"], r["odo"] / 10000) for r in cars]}[row["features"]]
        same("single r2 " + row["features"], ols(X, lylist)[1], row["r2_log"])
    print(f"check.py: {n_ok} recomputed numbers agree with results/")


if __name__ == "__main__":
    main()
