# DSN Mart Sales Prediction: Baseline and External Record Search

**Kazeem Mayowa** · DSN Bootcamp Qualification Hackathon 2026, ML Track

I wanted a comparison that shows what the DSN dataset supports by itself and what changes when the original Big Mart sales file is available. The notebook first trains a Ridge regression on the competition predictors. It then searches the labeled Big Mart records for nearby product-store examples and prepares a submission from that external reference.

## Files

| File | Use |
| --- | --- |
| `train.csv` | 6,818 DSN rows with `total_sales`. |
| `test.csv` | 1,705 DSN rows awaiting predictions. |
| `original_bigmart.csv` | Optional local copy of the original Big Mart training file with `Item_Outlet_Sales`. |
| `Kazeem_Mayowa_DSN_Mart_External_KNN.ipynb` | Data checks, DSN-only baseline, external search, error review, and export. |
| `kazeem_mayowa_submission.csv` | Created by the last notebook cell; columns `id,total_sales`. |

The external CSV is not included here. The notebook reads `original_bigmart.csv` if present or fetches the public file from the URL in its first code cell.

## The analysis

1. **Check the inputs.** Confirm shapes, target columns, and unique IDs. Review missing values and the high end of the sales distribution.
2. **Explore sales.** Compare median sales across product price bands and look at their spread by store location tier.
3. **Train a DSN-only reference model.** Split the labeled DSN data 80/20, fill missing features, encode categories, and score a Ridge regression against held-out DSN targets.
4. **Prepare source matching.** Normalize outlet age, location tier, and product category across the datasets. Fill product weight consistently and check how many source candidates each DSN row has.
5. **Search Big Mart rows.** Restrict candidates to the same age, tier, and category. Build a `cKDTree` from scaled price, weight, and visibility values and compare the sales of 1, 3, or 5 nearby records.
6. **Inspect and export.** Compare scores on the same DSN holdout rows, review absolute error by product category, then write the test predictions in their original ID order.

`product_code` helps fill missing DSN weights. Outlet age, tier, and category restrict the source search. Price, filled weight, and shelf visibility define distance within each candidate group. The plots and error table help explain the data and estimates; they are not extra target features.

## Measured local check

| Method | RMSE |
| --- | ---: |
| DSN-only Ridge regression | 1,122.35 |
| Big Mart source search, 1 neighbour | **209.87** |
| Big Mart source search, 3 neighbours | 378.56 |
| Big Mart source search, 5 neighbours | 457.58 |

Both sets of scores use the same fixed 20% DSN rows (`random_state=42`). The one-neighbour source search had the lowest RMSE and supplies the submission. These are local notebook results, not verified leaderboard results.

**The comparison has an important boundary:** Ridge is fitted only on the remaining DSN training labels. The Big Mart reference, by contrast, contains labeled counterparts of DSN train and test rows. Its search can draw on historical sales for counterparts of the held-out rows. That makes 209.87 a **source-assisted** check, not evidence of equal accuracy on new records without a labeled source counterpart. This notebook does not recreate the competition's exact seeded target transformation.

## Run the notebook

With Python 3.10 or newer:

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn notebook
python -m notebook Kazeem_Mayowa_DSN_Mart_External_KNN.ipynb
```

Keep `train.csv` and `test.csv` in the working directory with the notebook, or edit the paths in its first code cell. Put `original_bigmart.csv` there for an offline run; otherwise the configured public URL needs internet access. Run the cells in order. The final output is `kazeem_mayowa_submission.csv` with 1,705 rows.

## Takeaway

The DSN-only and external-source results answer different questions. The baseline estimates sales from competition features. The record search uses labels from an older public source with overlapping records. Reporting that distinction is part of reporting the result accurately.
