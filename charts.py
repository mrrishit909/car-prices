"""Step 6: charts.   python charts.py -> charts/00_cover.png, 01_cleaning.png, 02_depreciation.png, 03_honest.png, 04_features.png, 05_intervals.png"""
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
AU = pd.read_csv(R / "audit.csv"); DA = pd.read_csv(R / "dep_age.csv"); DM = pd.read_csv(R / "dep_miles.csv"); DR = pd.read_csv(R / "dep_rate.csv"); MO = pd.read_csv(R / "models.csv"); IM = pd.read_csv(R / "importance.csv")
IV = pd.read_csv(R / "intervals.csv"); IB = pd.read_csv(R / "interval_bands.csv"); SM = pd.read_csv(R / "summary.csv").set_index("key")["value"]
BRANDS = list(DA["make"].unique()); PAL = [BLUE, ORANGE, GREEN, "#c28ae0", "#e0b43a", "#5cc8d6", "#e06a8a", GRAY]


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def cleaning(ax):
    d = AU.iloc[1:].copy(); short = {"price is 0": "price is 0", "price 1 to 499": "price 1 to 499", "price above 150,000": "price over 150,000", "placeholder price (1234, 12345, 11111...)": "placeholder price",
                                      "model year missing or before 1995": "year missing or before 1995", "odometer missing": "odometer missing", "odometer above 400,000": "odometer over 400,000",
                                      "odometer under 100 on a 2019 or older car": "odometer under 100, old car", "make or model missing": "make or model missing", "title not clean (salvage, rebuilt, lien, missing...)": "title not clean",
                                      "re-listings of a car already counted": "re-listing of a car already counted"}
    y = np.arange(len(d))[::-1]; cols = [BLUE] * (len(d) - 1) + [ORANGE]; ax.barh(y, d["dropped"], color=cols, height=0.7)
    for yi, v in zip(y, d["dropped"]): ax.text(v + 1500, yi, f"{v:,}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([short[s] for s in d["step"]], fontsize=9); ax.set_xlim(0, 175000); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")); ax.grid(axis="y", visible=False); ax.set_xlabel("listings dropped at this step (each listing counted once, at the first rule it breaks)")
    ax.set_title(f"{int(SM['raw']):,} listings become {int(SM['cars']):,} cars", fontsize=12)


def age_curves(ax, labels=True):
    for b, c in zip(BRANDS, PAL):
        d = DA[DA["make"] == b]; ax.plot(d["age"], d["median_price"] / 1000, color=c, linewidth=2.2)
        if labels: ax.plot([], [], color=c, linewidth=2.2, label=b)
    if labels: ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper right", ncol=2)
    ax.set_xlim(0, 20.5); ax.set_ylim(0, None); ax.set_xlabel("age in years (2021 minus model year)"); ax.set_ylabel("median asking price ($ thousand)")


def miles_curves(ax):
    for lab, c in zip(("age 3-5", "age 6-9", "age 10-14"), (BLUE, ORANGE, GREEN)):
        d = DM[(DM["age_band"] == lab) & (DM["n"] >= 1000)]; x = (d["odo_from"] + np.minimum(d["odo_to"], 300000)) / 2 / 1000; ax.plot(x, d["median_price"] / 1000, color=c, linewidth=2.2, marker="o", markersize=4, label=lab)
    ax.set_xlabel("odometer, middle of the band (thousand miles)"); ax.set_ylabel("median asking price ($ thousand)"); ax.set_ylim(0, None); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")


def honest(axes):
    ax = axes[0]; m = MO[MO["model"] == "M4 boosting"].set_index("design"); order = ["D1 random listings (leaky)", "D2 by car, all listings", "D3 by car, one listing each", "D4 by make-model (unseen model)", "FWD later cars"]
    lab = ["random\nlistings", "folds\nby car", "one listing\nper car", "unseen\nmake-model", "later-posted\ncars"]; v = [m.loc[o, "r2_log"] for o in order]; x = np.arange(5)
    ax.bar(x, v, color=[ORANGE, GRAY, BLUE, BLUE, BLUE], width=0.62)
    for xi, vi in zip(x, v): ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5); ax.set_ylim(0, 1); ax.set_ylabel("R-squared of log price, boosting"); ax.grid(axis="x", visible=False); ax.set_title("Same model, five ways to score it", fontsize=12)
    ax = axes[1]; d = MO[MO["design"] == "D3 by car, one listing each"].reset_index(drop=True); x = np.arange(len(d)); v = 100 * d["mdape"]
    ax.bar(x, v, color=[GRAY, BLUE, BLUE, BLUE, GREEN], width=0.62)
    for xi, vi in zip(x, v): ax.text(xi, vi + 1, f"{vi:.0f}%", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["the\nmean", "age,\nmiles", "+ make,\nengine", "+ make-\nmodel", "boosting"], fontsize=8.5); ax.set_ylim(0, 65); ax.set_ylabel("typical miss (median % error)"); ax.grid(axis="x", visible=False); ax.set_title("Each added fact shrinks the miss", fontsize=12)


def features(axes):
    ax = axes[0]; d = IM.sort_values("rise"); y = np.arange(len(d)); ax.barh(y, d["rise"], color=BLUE, height=0.65)
    for yi, v in zip(y, d["rise"]): ax.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([s.replace(" (cylinders, fuel, transmission, drive)", "\n(cyl., fuel, trans., drive)") for s in d["feature"]], fontsize=8.5); ax.set_xlim(0, 0.42); ax.grid(axis="y", visible=False)
    ax.set_xlabel("rise in out-of-fold error when the feature is shuffled"); ax.set_title("What the boosted model leans on", fontsize=12)
    ax = axes[1]; d = DR.iloc[::-1].reset_index(drop=True); y = np.arange(len(d)); w = 0.38
    ax.barh(y + w / 2, -d["pct_per_year"], height=w * 0.92, color=BLUE, label="per year of age"); ax.barh(y - w / 2, -d["pct_per_10k_miles"], height=w * 0.92, color=ORANGE, label="per 10,000 miles")
    ax.set_yticks(y); ax.set_yticklabels(d["make"], fontsize=8.5); ax.set_xlim(0, 11.5); ax.set_xlabel("% of price lost (age and miles fitted together)"); ax.grid(axis="y", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper right"); ax.set_title("Depreciation by brand", fontsize=12)


def ranges(ax, fs=9):
    d = IB.iloc[::-1].reset_index(drop=True); y = np.arange(len(d)); ax.barh(y, (d["median_hi"] - d["median_lo"]) / 1000, left=d["median_lo"] / 1000, color=BLUE, height=0.55)
    for yi, lo, hi in zip(y, d["median_lo"], d["median_hi"]): ax.text(hi / 1000 + 1, yi, f"\\${lo / 1000:.1f}k to \\${hi / 1000:.1f}k", va="center", fontsize=fs, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(d["predicted_price"], fontsize=fs); ax.set_xlim(0, 105); ax.grid(axis="y", visible=False); ax.set_xlabel("typical 90% range of the asking price ($ thousand)")


def intervals(axes):
    ax = axes[0]; d = IB; x = np.arange(len(d)); ax.bar(x, 100 * d["coverage"], color=BLUE, width=0.62); ax.axhline(90, color=GREEN, linestyle=":", linewidth=1.3); ax.text(-0.45, 91, "90% promised", color=GREEN, fontsize=8.5, ha="left", va="bottom")
    for xi, v in zip(x, d["coverage"]): ax.text(xi, 100 * v + 1.5, f"{100 * v:.0f}%", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([s.replace(" and over", "+").replace("under ", "<") for s in d["predicted_price"]], fontsize=9); ax.set_ylim(0, 110); ax.set_ylabel("share of asking prices inside the range"); ax.set_xlabel("predicted price"); ax.grid(axis="x", visible=False)
    ax.set_title("Coverage by price band", fontsize=12); ranges(axes[1]); axes[1].set_title("What the 90% range looks like", fontsize=12)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    m4 = MO[(MO["model"] == "M4 boosting") & (MO["design"] == "D3 by car, one listing each")].iloc[0]; q = float(SM["q_d3_m4"]); lo, hi = 100 * (1 - np.exp(-q)), 100 * (np.exp(q) - 1)
    fig = plt.figure(figsize=(14, 8)); g = fig.add_gridspec(1, 2, width_ratios=[1.1, 1]); a = fig.add_subplot(g[0]); b = fig.add_subplot(g[1])
    age_curves(a); a.set_title("Median asking price by age, the 8 most listed brands", fontsize=14); ranges(b, 10.5); b.set_title("A 90% range for a car, by its predicted price", fontsize=14)
    fig.suptitle(f"A used car's asking price: a median {100 * m4['mdape']:.0f}% miss from age, mileage and model,\nand an honest 90% range from {lo:.0f}% below to {hi:.0f}% above the guess", x=0.01, ha="left", fontsize=19, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.3)); cleaning(ax); save(fig, "01_cleaning.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); age_curves(axes[0]); axes[0].set_title("Price by age", fontsize=12); miles_curves(axes[1]); axes[1].set_title("Price by mileage (bands of 1,000+ cars)", fontsize=12); save(fig, "02_depreciation.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); honest(axes); save(fig, "03_honest.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3)); features(axes); save(fig, "04_features.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8)); intervals(axes); save(fig, "05_intervals.png")


if __name__ == "__main__":
    main()
