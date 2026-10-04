"""Step 1: fetch the Craigslist used-vehicle listings from OpenML (dataset 43306) and keep a slim copy of the columns the analysis needs.

    python download.py   -> data/dataset_43306.pq (456 MB, deleted at the end), data/cars_slim.csv.gz, results/pull.json

Source: OpenML dataset 43306 "Used-Cars-Dataset" v2, https://www.openml.org/d/43306 (parquet at https://data.openml.org/datasets/0004/43306/dataset_43306.pq, the
download address OpenML's own API advertises). Licence recorded by OpenML: "CC0: Public Domain". The listings were scraped from Craigslist by Austin Reese (Kaggle
"Craigslist Cars And Trucks Data", https://github.com/AustinReese/UsedVehicleSearch) and re-published by OpenML; the CC0 label is the uploader's, Craigslist's
own terms of use restrict scraping, so nothing here is committed except small aggregate tables (data/ is gitignored). OpenML's metadata says collected 2020-12-03, the
posting_date column says 2021-04-04 to 2021-05-04; the analysis uses the dates in the data. Cite: Reese, A. (2021), Craigslist Cars & Trucks data, via OpenML #43306.
The slim file drops url, image_url, description, region_url and county (free text, links, empty); it keeps VIN (a vehicle number, used only to find re-listings).
"""
import hashlib
import json
import ssl
import time
import urllib.request
from datetime import date
from pathlib import Path

import certifi
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

HERE = Path(__file__).parent
URL = "https://data.openml.org/datasets/0004/43306/dataset_43306.pq"
RAW = HERE / "data" / "dataset_43306.pq"
SLIM = HERE / "data" / "cars_slim.csv.gz"
COLS = ["id", "region", "price", "year", "manufacturer", "model", "condition", "cylinders", "fuel", "odometer", "title_status", "transmission", "VIN", "drive", "size", "type", "paint_color", "state", "lat", "long", "posting_date"]
CTX = ssl.create_default_context(cafile=certifi.where())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    if not RAW.exists():
        for attempt in range(4):
            try:
                req = urllib.request.Request(URL, headers={"User-Agent": "research-portfolio (Rishit Raj Mathur)"})
                with urllib.request.urlopen(req, context=CTX, timeout=120) as r, open(RAW, "wb") as out:
                    while chunk := r.read(1 << 20):
                        out.write(chunk)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(5)
    t = pq.read_table(RAW, columns=COLS)
    import gzip
    with gzip.open(SLIM, "wb") as f:
        pacsv.write_csv(t, f)
    info = {"pulled": date.today().isoformat(), "rows": t.num_rows,
            "files": {"dataset_43306.pq": {"url": URL, "bytes": RAW.stat().st_size, "sha256": sha(RAW)},
                      "cars_slim.csv.gz": {"url": "derived from dataset_43306.pq: columns " + ",".join(COLS), "bytes": SLIM.stat().st_size, "sha256": sha(SLIM)}}}
    (HERE / "results" / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    RAW.unlink()
    print(json.dumps({k: v["bytes"] for k, v in info["files"].items()}), info["rows"], "rows")


if __name__ == "__main__":
    main()
