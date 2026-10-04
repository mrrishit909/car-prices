# How accurately can a used car's price be predicted?

426,880 real Craigslist used-vehicle listings (US, posted 4 April to 4 May 2021), with asking prices. Five questions:
- how much of a listing's price do age, mileage, make-model and engine explain, and how large is the typical miss;
- how much an ordinary evaluation flatters the answer when the same car is listed several times (it is: 52% of listings have a twin in another random fold) and when a model is new to the fit;
- how fast a car loses value with age and with miles, brand by brand, and which price outliers and typos have to go (and how many);
- which features matter, with permutation importance measured out of fold;
- how wide an honest 90% interval is in dollars.

The task follows GeeksforGeeks' "Car Price Prediction Analysis". The page with the results and the charts is the build log:

Build log: https://mrrishit909.github.io/projects/car-prices/

## Data

OpenML dataset 43306, "Used-Cars-Dataset" (https://www.openml.org/d/43306; parquet at https://data.openml.org/datasets/0004/43306/dataset_43306.pq, 456 MB). Licence recorded by OpenML: CC0 Public Domain. The listings were scraped from Craigslist by Austin Reese (https://github.com/AustinReese/UsedVehicleSearch, also on Kaggle as "Craigslist Cars And Trucks Data") and re-published by OpenML. The CC0 label is the uploader's; Craigslist's own terms of use restrict scraping, so this repository commits no listing data, only small aggregate tables (`data/` is gitignored). OpenML's metadata says the data was collected on 2020-12-03, but the posting dates in the file run from 2021-04-04 to 2021-05-04; the analysis goes by the dates in the data. `download.py` keeps a 17 MB slim file (it drops url, image_url, description and other free text). Cite: Reese, A. (2021), Craigslist Cars & Trucks data, via OpenML #43306.

```
./venv/bin/pip install pandas numpy scipy scikit-learn matplotlib certifi pyarrow      # in a venv
./venv/bin/python download.py    # data/cars_slim.csv.gz, results/pull.json (downloads 456 MB, then deletes it)
./venv/bin/python analyze.py     # results/*, data/oof_*.csv.gz   (about 3 minutes)
./venv/bin/python check.py       # plain Python, about 6 minutes
./venv/bin/python charts.py
```

## Steps

1. **Download** the parquet file, checksum it, keep the columns the analysis needs.
2. **Audit and clean.** Each listing is dropped for the first rule it breaks: price 0; price 1 to 499; price over 150,000; a placeholder price (1234, 12345, 11111...); model year missing or before 1995; odometer missing, over 400,000, or under 100 on a car of model year 2019 or older; make or model missing; title not clean. Age is 2021 minus the model year.
3. **Find the re-listings.** Two listings are the same car when they share a usable VIN (17 characters, no I, O or Q, not one repeated character, and all its listings agree on make and year) or the same make, model, year and a non-round odometer. The cars are the connected components; each keeps its lowest id.
4. **Evaluate five ways**, 5 folds each, fold = md5 of the group key mod 5: D1 random listings (leaky); D2 folds by car, all listings; D3 folds by car, one listing per car (the headline); D4 folds by make-model (a model the fit has never seen); FWD train on cars first posted up to 2021-04-26, test on cars first posted later. Models, all on log price: M0 the training mean; M1 OLS on age, age squared, miles, miles squared; M2 ridge adding make, cylinders, fuel, transmission and drive; M3 ridge adding the 250 most common make-models; M4 gradient boosting (sklearn, fixed settings, not tuned) on those and body type, condition and state. The ridge fits are the normal equations with lambda = 100.
5. **Depreciation, importance, intervals.** Median price by age and by odometer band; log price on age and miles by brand; permutation importance on M4 (shuffle a feature group inside each test fold, out of fold); split-conformal 90% intervals on the log scale, with the half-width for each fold taken from the other folds' absolute residuals.
6. **Check.** `check.py` recomputes everything from the downloaded file in plain Python (csv, gzip, hashlib, math and loops: no pandas, numpy, scipy or sklearn): 719 numbers, to a relative tolerance of 1e-8. The ridge fits are solved by Cholesky from per-fold sums; the boosting predictions are stored out of fold and every metric built from them is recomputed, after the stored rows are tied to the recomputed cars and folds.
7. **Charts.**

## Results

- **Cleaning.** 426,880 listings, 330,469 after the rules (77.4%), 188,938 cars after the re-listings. 32,895 listings (7.7% of the raw file) ask $0 and 9,199 ask $1 to $499. 141,531 clean listings (42.8%) are re-listings of a car already counted; the most listed car appears 261 times and 38,241 cars are listed more than once. The VIN alone would give 208,869 cars; the make-model-year-odometer rule joins 19,931 more. The median car asks $11,720 (mean $15,598), is 9 years old and has 107,000 miles.
- **Depreciation.** With age and miles fitted together a car loses 7.1% of its asking price per year and 3.5% per 10,000 miles. Age and miles are correlated (0.63), which is why they share the work: R-squared of log price is 0.40 on age alone, 0.30 on miles alone and 0.44 on both. By brand (the 12 most listed) the yearly loss runs from 4.4% (jeep) to 8.4% (hyundai) and the loss per 10,000 miles from 0.6% (ram) to 6.4% (bmw). A Toyota's median asking price is $31,790 at one year, $20,399 at five, $9,800 at ten and $6,495 at fifteen.
- **Each added fact helps.** On one listing per car (D3) the typical miss (median absolute percentage error) is 56% for the mean, 35% with age and miles, 23% adding make and engine, 21% adding the make-model, and 15% for boosting; the R-squared of log price is -0.00, 0.45, 0.62, 0.65 and 0.79. Boosting lands within 10% of the asking price for 34% of cars and within 20% for 61%; its mean absolute error is $3,151.
- **The way it is scored changes the answer.** The same boosted model has an R-squared of 0.84 (typical miss 12.0%) on random listings, 0.82 (13.5%) when folds are by car, 0.79 (15.4%) with one listing per car, 0.73 (18.4%) on a make-model it has never seen and 0.72 (16.8%) on cars first posted after 2021-04-26. The gap between random listings and by-car folds is modest for this model, whose settings I did not vary, so I cannot say whether a more flexible model would exploit the twins more. A make-model the fit has never seen raises the typical miss by 3.3 points for the ridge fit and 3.0 points for boosting.
- **What matters (permutation importance, out of fold).** Shuffling age raises the out-of-fold RMSE of log price by 0.34, the make and model together by 0.19, mileage by 0.14, the engine fields by 0.12, state by 0.07, body type by 0.04 and condition by 0.04. Age and mileage are correlated, so the two numbers are not additive shares.
- **Intervals.** The 90% interval for boosting runs from 0.60 times to 1.66 times the prediction (log half-width 0.508): $11,813 wide for the median car, against $14,774 for the ridge fit and $23,649 for age and miles alone. Overall coverage is 90.0%, but it varies with the price: 81% under $5k, 89% from $5k to $10k, 92%, 95% and 95% above $40k. An interval calibrated on random-fold residuals covers only 85.3% of new cars, and the correct one covers 86.9% of cars from unseen make-models.

## Not done, and caveats

- The price is the asking price on Craigslist, not what the car sold for; a seller can ask more or less than a buyer pays, and the data has no sale price.
- One month, spring 2021, in the US only; prices in other months or countries may differ, and the forward split (train on earlier-posted cars, test on later ones, 97,882 and 91,056 cars) is a split inside that month, not a test of next year.
- The listings are a selection: cars whose sellers use Craigslist, dealers and private sellers mixed (the file has no seller type), and only cars that pass the cleaning rules (clean title, price $500 to $150,000, model year 1995 or later, odometer present). 35% of clean listings have no usable VIN, so some twins that are not caught by the make-model-year-odometer rule remain, and the by-car designs still leak a little; the same rule could also join two different cars with an identical odometer reading.
- The boosting model uses fixed settings and was not tuned; the depreciation rates are associations in listings (age and miles are correlated and sellers choose what to write), not causes. Permutation importance with correlated features is approximate.
- The conformal interval has the promised coverage on average over cars, not for every kind of car (see the coverage by price). Ridge fits and the boosting predictions are of log price, so predictions are medians, not means.
- The CC0 label is the uploader's and Craigslist's terms restrict scraping; nothing from the listings is committed, only summaries.
