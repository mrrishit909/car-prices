"""Steps 2-4: clean the listings, find the re-listings, evaluate price models the honest way, draw the depreciation curves.

    python analyze.py   -> results/*.csv (small aggregates), data/oof_*.csv.gz (out-of-fold predictions, not committed)

Data: Craigslist US used-vehicle listings posted 2021-04-04 to 2021-05-04 (OpenML 43306, CC0, see download.py). Price is the ASKING price in US dollars.
Cleaning (each listing is dropped for the FIRST rule it breaks, in this order): price 0; price 1-499; price above 150,000; a placeholder price (1234, 12345, 123456 or a
price of 5+ digits that is one repeated digit); model year missing or before 1995; odometer missing; odometer above 400,000; odometer under 100 on a car of model year 2019
or older; make or model missing; title status missing or not "clean". age = max(0, 2021 - model year), odometer in miles.
Re-listings: two listings are the same CAR when they share a usable VIN (17 characters, no I/O/Q, not one repeated character, and all its listings agree on make and year),
or share (make, model, year, odometer) with an odometer that is not a multiple of 1000 and not a digit pattern. Cars are the connected components; the car keeps its lowest id.
Evaluation designs, 5 folds each, fold = md5(group key) mod 5: D1 random listings (leaky: a car's twins sit on both sides); D2 folds by car, all clean listings;
D3 folds by car, one listing per car (the HEADLINE design); D4 folds by make+model on one listing per car (a model the fit has never seen); FWD train on cars first posted
up to 2021-04-26, test on cars first posted later (the nearest thing to "only from the past" a one-month sample allows).
Models, all on log price: M0 the training mean; M1 OLS on age, age^2, miles, miles^2; M2 ridge = M1 + make + cylinders + fuel + transmission + drive; M3 ridge = M2 + 250 most
common make-models; M4 gradient boosting (sklearn HistGradientBoostingRegressor, fixed settings, not tuned) on age, miles, make, make-model, cylinders, fuel, transmission,
drive, body type, condition, state. The ridge models are the normal equations (X'X + lambda D) b = X'y with lambda = 100 on every column but the intercept, columns scaled
(age/10, miles/100000), one reference level dropped per categorical. Out-of-fold predictions of M4 are stored and every metric recomputed from them by check.py.
Intervals: split-conformal across folds on the log scale: for fold k the half-width is the ceil((n+1)*0.9)-th smallest absolute residual of the other folds.
Permutation importance: in each test fold a feature group is shuffled (rows permuted jointly) and the rise in out-of-fold RMSE of log price is the importance.
"""
import hashlib
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
K, LAM, TOPK, CUT, REFYEAR = 5, 100.0, 250, "2021-04-26", 2021
VIN_RE = re.compile(r"[A-HJ-NPR-Z0-9]{17}")
CATS = ["cylinders", "fuel", "transmission", "drive", "type", "condition", "state"]
PERM_GROUPS = {"age": ["age"], "mileage": ["odo"], "make + model": ["make", "model_key"], "engine (cylinders, fuel, transmission, drive)": ["cylinders", "fuel", "transmission", "drive"],
               "body type": ["type"], "condition": ["condition"], "state": ["state"]}


def norm(s):
    return " ".join(str(s).lower().split())


def load():
    df = pd.read_csv(D / "cars_slim.csv.gz", keep_default_na=False, na_values=[""], dtype={"VIN": str})
    for c in ["manufacturer", "model", "title_status"] + CATS:
        df[c] = df[c].map(lambda s: norm(s) if isinstance(s, str) else s).replace("", np.nan)
    return df


def placeholder_price(p):
    t = str(int(p))
    return p in (1234, 12345, 123456, 1234567) or (len(t) >= 5 and len(set(t)) == 1)


def clean(df):
    rules = [("price is 0", df.price == 0),
             ("price 1 to 499", (df.price > 0) & (df.price < 500)),
             ("price above 150,000", df.price > 150000),
             ("placeholder price (1234, 12345, 11111...)", df.price.map(placeholder_price)),
             ("model year missing or before 1995", df.year.isna() | (df.year < 1995)),
             ("odometer missing", df.odometer.isna()),
             ("odometer above 400,000", df.odometer > 400000),
             ("odometer under 100 on a 2019 or older car", (df.odometer < 100) & (df.year <= 2019)),
             ("make or model missing", df.manufacturer.isna() | df.model.isna()),
             ("title not clean (salvage, rebuilt, lien, missing...)", df.title_status.isna() | (df.title_status != "clean"))]
    alive = np.ones(len(df), bool); rows = [("raw listings", 0, len(df))]
    for name, mask in rules:
        hit = alive & mask.to_numpy(); alive &= ~hit; rows.append((name, int(hit.sum()), int(alive.sum())))
    return df[alive].copy(), rows


def is_pattern_odo(o):
    t = str(int(o))
    return o % 1000 == 0 or len(set(t)) == 1 or t in ("1234", "12345", "123456", "1234567")


def add_groups(c):
    c = c.reset_index(drop=True)
    c["make"] = c.manufacturer; c["mstr"] = c.model; c["age"] = (REFYEAR - c.year).clip(lower=0); c["odo"] = c.odometer
    vin = c.VIN.map(lambda s: s.strip().upper() if isinstance(s, str) else None)
    ok = vin.map(lambda s: isinstance(s, str) and VIN_RE.fullmatch(s) is not None and len(set(s)) > 1)
    vin = vin.where(ok)
    my = (c.make + "|" + c.year.astype(int).astype(str)).where(vin.notna())
    nun = my.groupby(vin).nunique()
    bad = set(nun.index[nun > 1])
    vin_key = vin.where(~vin.isin(bad))
    near = (c.make + "|" + c.mstr + "|" + c.year.astype(int).astype(str) + "|" + c.odo.astype(int).astype(str)).where(~c.odo.map(is_pattern_odo))
    src, dst = [], []
    for key in (vin_key, near):
        k = key.dropna(); first = k.index.to_series().groupby(k).transform("first")
        src += list(first.index); dst += list(first.values)
    n = len(c)
    g = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    c["comp"] = lab
    c["car"] = c.groupby("comp").id.transform("min")
    c["day"] = c.posting_date.str[:10]; c["first_date"] = c.groupby("comp").day.transform("min")
    info = {"vin_valid_listings": int(vin.notna().sum()), "vin_distinct_valid": int(vin.nunique()), "vin_unreliable": len(bad), "vin_unreliable_listings": int(vin.isin(bad).sum()),
            "near_key_listings": int(near.notna().sum()), "cars": ncomp, "largest_car_listings": int(c.groupby("comp").size().max()),
            "listings_in_multi": int((c.groupby("comp").id.transform("size") > 1).sum()), "cars_listed_more_than_once": int((c.groupby("comp").size() > 1).sum()),
            "top_vin_listings": int(vin.value_counts().iloc[0])}
    # near-key links that join two groups the VIN rule alone kept apart
    info["cars_by_vin_rule_only"] = int(vin_key.where(vin_key.notna(), "#" + c.index.astype(str)).nunique())
    return c, info


def fold_of(key):
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16) % K


def conf_q(absres):
    s = np.sort(absres); n = len(s); return s[((n + 1) * 9 + 9) // 10 - 1]


def metrics(ly, p, name):
    price, pred = np.exp(ly), np.exp(p); e = p - ly; ape = np.abs(pred - price) / price; sst = ((ly - ly.mean()) ** 2).sum()
    return {"model": name, "n": len(ly), "rmse_log": math.sqrt((e ** 2).mean()), "r2_log": 1 - (e ** 2).sum() / sst, "mae_usd": np.abs(pred - price).mean(), "mdape": np.median(ape),
            "within10": (np.abs(pred / price - 1) <= 0.10).mean(), "within20": (np.abs(pred / price - 1) <= 0.20).mean()}


class Features:
    def __init__(self, cl, cars):
        d = cars
        cnt = d.groupby(["make", "mstr"]).size().reset_index(name="n").sort_values(["n", "make", "mstr"], ascending=[False, True, True]).head(TOPK)
        self.keys = [f"{a}|{b}" for a, b in zip(cnt.make, cnt.mstr)]
        self.lev = {c: sorted(cl[c].fillna("unk").unique()) for c in CATS + ["make"]}
        self.lev["model_key"] = sorted(self.keys) + ["other"]

    def prep(self, d):
        d = d.copy(); d["model_key"] = (d.make + "|" + d.mstr).where((d.make + "|" + d.mstr).isin(set(self.keys)), "other")
        for c in CATS: d[c] = d[c].fillna("unk")
        return d

    def ridge_X(self, d, level):          # level 1: M1, 2: M2, 3: M3 (d already prepped)
        n = len(d); a, m = d.age.to_numpy() / 10, d.odo.to_numpy() / 100000
        cols = [np.ones(n), a, a * a, m, m * m]; blocks = [sp.csr_matrix(np.column_stack(cols))]
        names = (["make", "cylinders", "fuel", "transmission", "drive"] if level >= 2 else []) + (["model_key"] if level >= 3 else [])
        for c in names:
            levs = self.lev[c][1:] if c != "model_key" else self.lev[c][:-1]      # reference level: first alphabetically ("other" for model keys)
            idx = {l: i for i, l in enumerate(levs)}; v = d[c].map(idx)
            ok = v.notna().to_numpy()
            blocks.append(sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), v[ok].astype(int).to_numpy())), shape=(n, len(levs))))
        return sp.hstack(blocks, format="csr")

    def hgb_X(self, d):
        cols = [d.age.to_numpy(), d.odo.to_numpy()]
        for c in ["make", "model_key", "cylinders", "fuel", "transmission", "drive", "type", "condition", "state"]:
            idx = {l: i for i, l in enumerate(self.lev[c])}; cols.append(d[c].map(idx).fillna(-1).to_numpy())
        names = ["age", "odo", "make", "model_key", "cylinders", "fuel", "transmission", "drive", "type", "condition", "state"]
        return np.column_stack(cols).astype(float), names


def ridge_fit_predict(F, tr, te, level):
    X, Xt = F.ridge_X(tr, level), F.ridge_X(te, level); y = np.log(tr.price.to_numpy())
    A = (X.T @ X).toarray(); pen = np.full(A.shape[0], LAM); pen[0] = 0; A[np.diag_indices_from(A)] += pen
    b = np.linalg.solve(A, X.T @ y); return Xt @ b


def hgb_fit(F, tr):
    X, names = F.hgb_X(tr); cat = [n in ("make", "model_key", "cylinders", "fuel", "transmission", "drive", "type", "condition", "state") for n in names]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=40, early_stopping=False, categorical_features=cat, random_state=0)
    return m.fit(X, np.log(tr.price.to_numpy()))


def run_design(name, d, key, F, models, perm=False, seed=0):
    """5 folds by hash of `key`; out-of-fold predictions for each model; returns the frame."""
    d = d.copy(); d["fold"] = d[key].map(fold_of); out = d[["id", "car", "fold", "price"]].copy(); out["ly"] = np.log(d.price.to_numpy())
    for mname in models: out[mname] = np.nan
    if perm:
        for g in PERM_GROUPS: out["perm: " + g] = np.nan
    rng = np.random.default_rng(seed)
    for k in range(K):
        tr, te = d[d.fold != k], d[d.fold == k]; ti = te.index
        if "m0" in models: out.loc[ti, "m0"] = np.log(tr.price.to_numpy()).mean()
        for lv, mn in ((1, "m1"), (2, "m2"), (3, "m3")):
            if mn in models: out.loc[ti, mn] = ridge_fit_predict(F, tr, te, lv)
        if "m4" in models:
            mod = hgb_fit(F, tr); X, names = F.hgb_X(te); out.loc[ti, "m4"] = mod.predict(X)
            if perm:
                for g, cs in PERM_GROUPS.items():
                    Xp = X.copy(); pi = rng.permutation(len(te)); ix = [names.index(c) for c in cs]; Xp[:, ix] = X[pi][:, ix]; out.loc[ti, "perm: " + g] = mod.predict(Xp)
        print(name, "fold", k, len(te), flush=True)
    return out


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X]); b = np.linalg.solve(X1.T @ X1, X1.T @ y); e = y - X1 @ b
    return b, 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    R.mkdir(exist_ok=True)
    raw = load(); cl, steps = clean(raw)
    audit = pd.DataFrame(steps, columns=["step", "dropped", "left"]); cl, ginfo = add_groups(cl)
    cars = cl[cl.id == cl.car].copy().reset_index(drop=True)
    audit.loc[len(audit)] = ["re-listings of a car already counted", len(cl) - len(cars), len(cars)]
    audit.to_csv(R / "audit.csv", index=False)
    assert cl.id.is_unique
    F = Features(cl, cars); cl, cars = F.prep(cl), F.prep(cars)
    ly_all = np.log(cars.price.to_numpy())
    summary = dict(ginfo); summary.update({"raw": len(raw), "clean_listings": len(cl), "cars": len(cars), "model_keys": len(F.keys), "makes": len(F.lev["make"]), "model_keys_share_of_cars": float(cars.model_key.ne("other").mean()),
                                           "median_price": float(cars.price.median()), "mean_price": float(cars.price.mean()), "median_age": float(cars.age.median()), "median_odo": float(cars.odo.median()),
                                           "corr_age_odo": float(np.corrcoef(cars.age, cars.odo)[0, 1]), "first_post": cl.posting_date.str[:10].min(), "last_post": cl.posting_date.str[:10].max(),
                                           "lambda": LAM})
    # ---- evaluation designs
    cl["g1"] = cl.id.astype(str); cl["g2"] = cl.car.astype(str); cars["g3"] = cars.car.astype(str); cars["g4"] = cars.make + "|" + cars.mstr
    ladder = ["m0", "m1", "m2", "m3", "m4"]
    o1 = run_design("D1", cl, "g1", F, ["m4"]); o2 = run_design("D2", cl, "g2", F, ["m4"])
    o3 = run_design("D3", cars, "g3", F, ladder, perm=True, seed=1); o4 = run_design("D4", cars, "g4", F, ladder)
    # forward split
    tr, te = cars[cars.first_date <= CUT], cars[cars.first_date > CUT]; fw = te[["id", "car", "price"]].copy(); fw["ly"] = np.log(te.price.to_numpy()); fw["fold"] = 0
    fw["m0"] = np.log(tr.price.to_numpy()).mean()
    for lv, mn in ((1, "m1"), (2, "m2"), (3, "m3")): fw[mn] = ridge_fit_predict(F, tr, te, lv)
    mod = hgb_fit(F, tr); X, _ = F.hgb_X(te); fw["m4"] = mod.predict(X)
    for nm, o in (("d1", o1), ("d2", o2), ("d3", o3), ("d4", o4), ("fwd", fw)): o.to_csv(D / f"oof_{nm}.csv.gz", index=False, float_format="%.17g")
    summary.update({"fwd_train": len(tr), "fwd_test": len(te), "fwd_cut": CUT})
    # twins in the other folds under random listing folds
    cl["fold1"] = cl.g1.map(fold_of); has_other = cl.groupby("car").fold1.transform("nunique") > 1
    # a listing has a twin in training when its car has a listing in some other fold
    summary["d1_test_listings_with_twin_in_train"] = float(has_other.mean())
    names = {"m0": "M0 the mean", "m1": "M1 OLS: age, miles", "m2": "M2 ridge: + make, engine", "m3": "M3 ridge: + make-model", "m4": "M4 boosting"}
    rows = []
    for dn, o, ms in (("D1 random listings (leaky)", o1, ["m4"]), ("D2 by car, all listings", o2, ["m4"]), ("D3 by car, one listing each", o3, ladder), ("D4 by make-model (unseen model)", o4, ladder), ("FWD later cars", fw, ladder)):
        for m in ms: rows.append({"design": dn, **metrics(o.ly.to_numpy(), o[m].to_numpy(), names[m])})
    pd.DataFrame(rows).to_csv(R / "models.csv", index=False)
    folds = []
    for dn, o in (("D1", o1), ("D2", o2), ("D3", o3), ("D4", o4)):
        for k in range(K): s = o[o.fold == k]; folds.append({"design": dn, "fold": k, "rows": len(s), "cars": s.car.nunique()})
    pd.DataFrame(folds).to_csv(R / "folds.csv", index=False)
    # ---- permutation importance (D3, M4)
    base = metrics(o3.ly.to_numpy(), o3.m4.to_numpy(), "")["rmse_log"]; imp = []
    for g in PERM_GROUPS:
        r = metrics(o3.ly.to_numpy(), o3["perm: " + g].to_numpy(), "")["rmse_log"]; imp.append({"feature": g, "rmse_log_shuffled": r, "rmse_log_base": base, "rise": r - base})
    imp = pd.DataFrame(imp); imp["rise_r"] = imp.rise.round(6); imp = imp.sort_values(["rise_r", "feature"], ascending=[False, True]).drop(columns="rise_r"); imp.to_csv(R / "importance.csv", index=False)
    # ---- intervals (conformal across folds)
    rows, bands = [], []
    for nm, o, mcol in (("M1 OLS: age, miles", o3, "m1"), ("M3 ridge: + make-model", o3, "m3"), ("M4 boosting", o3, "m4")):
        res = (o.ly - o[mcol]).to_numpy(); qk = {k: conf_q(np.abs(res[(o.fold != k).to_numpy()])) for k in range(K)}
        q = o.fold.map(qk).to_numpy(); cov = np.abs(res) <= q; pred = np.exp(o[mcol].to_numpy()); lo, hi = pred * np.exp(-q), pred * np.exp(q)
        qa = conf_q(np.abs(res))
        rows.append({"model": nm, "scope": "D3 cars, folds by car", "n": len(o), "coverage": cov.mean(), "q_log": qa, "median_width_usd": np.median(hi - lo), "median_relative_halfwidth": float(np.median((hi - lo) / 2 / pred))})
        if mcol == "m4":
            for lab, a, b in (("under $5k", 0, 5000), ("$5k-10k", 5000, 10000), ("$10k-20k", 10000, 20000), ("$20k-40k", 20000, 40000), ("$40k and over", 40000, 1e12)):
                s = (pred >= a) & (pred < b); bands.append({"predicted_price": lab, "n": int(s.sum()), "coverage": cov[s].mean(), "median_width_usd": np.median((hi - lo)[s]), "median_lo": np.median(lo[s]), "median_hi": np.median(hi[s])})
            r1 = (o1.ly - o1.m4).to_numpy(); ql = conf_q(np.abs(r1))
            rows.append({"model": nm, "scope": "interval calibrated on D1 random folds, tried on D3 cars", "n": len(o), "coverage": float((np.abs(res) <= ql).mean()), "q_log": ql, "median_width_usd": float(np.median(pred * (np.exp(ql) - np.exp(-ql)))), "median_relative_halfwidth": (math.exp(ql) - math.exp(-ql)) / 2})
            r4 = (o4.ly - o4.m4).to_numpy(); p4 = np.exp(o4.m4.to_numpy())
            rows.append({"model": nm, "scope": "interval calibrated on D3, tried on D4 unseen models", "n": len(o4), "coverage": float((np.abs(r4) <= qa).mean()), "q_log": qa, "median_width_usd": float(np.median(p4 * (np.exp(qa) - np.exp(-qa)))), "median_relative_halfwidth": (math.exp(qa) - math.exp(-qa)) / 2})
            summary["q_d3_m4"] = qa
    pd.DataFrame(rows).to_csv(R / "intervals.csv", index=False); pd.DataFrame(bands).to_csv(R / "interval_bands.csv", index=False)
    # ---- depreciation
    top = cars.groupby("make").size().reset_index(name="n").sort_values(["n", "make"], ascending=[False, True]); top8 = list(top.make.head(8)); top12 = list(top.make.head(12))
    da = []
    for mk in top8:
        s = cars[cars.make == mk]
        for a in range(0, 21):
            t = s[s.age == a]
            if len(t) >= 30: da.append({"make": mk, "age": a, "n": len(t), "median_price": float(t.price.median())})
    pd.DataFrame(da).to_csv(R / "dep_age.csv", index=False)
    edges = [0, 20000, 40000, 60000, 80000, 100000, 125000, 150000, 200000, 300000, 400001]; dm = []
    for lab, lo_a, hi_a in (("age 3-5", 3, 5), ("age 6-9", 6, 9), ("age 10-14", 10, 14)):
        s = cars[(cars.age >= lo_a) & (cars.age <= hi_a)]
        for i in range(len(edges) - 1):
            t = s[(s.odo >= edges[i]) & (s.odo < edges[i + 1])]
            if len(t) >= 30: dm.append({"age_band": lab, "odo_from": edges[i], "odo_to": edges[i + 1], "n": len(t), "median_price": float(t.price.median())})
    pd.DataFrame(dm).to_csv(R / "dep_miles.csv", index=False)
    dr = []
    for mk in ["all makes"] + top12:
        s = cars if mk == "all makes" else cars[cars.make == mk]
        b, r2 = ols(np.column_stack([s.age.to_numpy(), s.odo.to_numpy() / 10000]), np.log(s.price.to_numpy()))
        dr.append({"make": mk, "n": len(s), "pct_per_year": 100 * (math.exp(b[1]) - 1), "pct_per_10k_miles": 100 * (math.exp(b[2]) - 1), "r2_log": r2})
    pd.DataFrame(dr).to_csv(R / "dep_rate.csv", index=False)
    sf = []
    for lab, X in (("age only", cars[["age"]].to_numpy()), ("miles only", cars[["odo"]].to_numpy() / 10000), ("age and miles", np.column_stack([cars.age, cars.odo / 10000]))):
        b, r2 = ols(X, ly_all); sf.append({"features": lab, "r2_log": r2})
    pd.DataFrame(sf).to_csv(R / "single_r2.csv", index=False)
    pd.DataFrame([{"key": k, "value": v} for k, v in summary.items()]).to_csv(R / "summary.csv", index=False)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
