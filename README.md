# 💡 Lumen — Automated Data Analysis Dashboard

Upload a data file and Lumen instantly builds a dashboard — headline numbers,
plain-English findings, automatic charts, and a **"records to review" anomaly
check** that surfaces unusual rows (the general form of fraud / error detection).
Power tools for cleaning, SQL, prediction, and reporting are one click away.

**No spreadsheets. No code. Just upload and read the insights.**

---

## ✨ What it does

The moment you load data, the **Dashboard** auto-generates:

- **At-a-glance KPIs** — records, fields, data completeness, duplicates, and the total of any money/amount column it detects.
- **Key findings in plain English** — e.g. *"About 4% of values are missing; total_charges is the most incomplete"*, *"Month-to-month is the most common contract (55%)"*, *"tenure and total charges rise together (0.83)"*.
- **🚩 Records to review** — an Isolation Forest flags rows that look unusual vs. the rest, with a sensitivity slider and a one-click CSV export. This is the dataset-agnostic version of fraud/error screening (claims, transactions, billing…).
- **Automatic charts** — Lumen picks sensible visuals from your columns: trends over time (if a date exists), category breakdowns, distributions, and a correlation heatmap.

Then, in the top tabs:

| Tab | What it's for |
|---|---|
| 🔎 **Explore** | Build your own chart in a couple of clicks; see what moves together. |
| 🧹 **Clean up** | Remove duplicates, fill blanks, drop columns, cap extreme values — each step is recorded and **exportable as a reproducible pandas script**. |
| 🗄️ **Ask in SQL** | Query your data with real SQL (read-only sandbox). |
| 🔮 **Predict** | Pick what you want to predict; Lumen trains several models and shows accuracy + which factors matter most. |
| 📄 **Report** | Export a polished, self-contained HTML report (print to PDF). |

---

## 🧰 Tech stack

**Language:** Python 3.10+
**App / UI:** Streamlit (single-page tabbed dashboard)
**Data wrangling:** pandas, NumPy
**Visualization:** Plotly
**Machine learning / anomaly detection:** scikit-learn (Pipeline, ColumnTransformer, Isolation Forest, cross-validation)
**Statistics:** SciPy, statsmodels
**Database / SQL:** SQLAlchemy, SQLite
**Reporting:** Jinja2 (templated HTML)
**File formats:** CSV, Excel (openpyxl), Parquet (pyarrow), JSON
**Testing:** pytest

---

## 🚀 Quickstart

```bash
git clone https://github.com/woooj02/Lumen.git lumen && cd lumen
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run.sh                      # or: streamlit run app.py
```

Open the URL it prints (default http://localhost:8501), click
**"Try the sample dataset"**, and the dashboard builds itself.

Run the tests:

```bash
pytest -q
```

---

## 📁 Project structure

```
lumen/
├── app.py                  # single-page dashboard + tabs (thin UI layer)
├── src/                    # reusable, unit-tested analytics engine
│   ├── data_loader.py      # multi-format ingestion + dtype/date handling
│   ├── profiling.py        # automated EDA & quality flags
│   ├── insights.py         # auto-KPIs, plain-English findings, smart chart picks
│   ├── anomaly.py          # Isolation Forest "records to review" detection
│   ├── cleaning.py         # transforms that emit reproducible pandas code
│   ├── analysis.py         # correlations + hypothesis tests (scipy)
│   ├── viz.py              # Plotly chart builders
│   ├── modeling.py         # AutoML pipeline + model comparison (sklearn)
│   ├── db.py               # SQL query engine (SQLAlchemy/SQLite)
│   ├── report.py           # Jinja2 HTML report generator
│   ├── sample_data.py      # synthetic churn dataset generator
│   └── state.py            # Streamlit session-state helpers
├── tests/test_core.py      # pytest coverage of the analytics engine
├── requirements.txt
└── .streamlit/config.toml
```

---

## 🧠 Design highlights (worth talking about in an interview)

- **Auto-insight engine:** `insights.py` chooses KPIs, narrative findings, and charts from the data's shape — no manual configuration, which is what makes it usable by non-analysts.
- **General anomaly detection:** an Isolation Forest over scaled numeric + encoded categorical features flags outliers on *any* dataset, and reports which fields most separate flagged rows from normal ones.
- **No data leakage in modelling:** all preprocessing (impute → encode → scale) is wrapped in a scikit-learn `Pipeline` + `ColumnTransformer`, fit only on training folds during cross-validation.
- **Reproducible cleaning:** point-and-click cleaning records each step's pandas code, downloadable as a runnable `cleaning_pipeline.py` — clicks become a version-controllable ETL script.
- **Separation of concerns:** every capability lives in `src/` as a pure, `pytest`-covered function; the Streamlit app is a thin presentation layer.

---

## 📊 Sample dataset

`src/sample_data.py` generates a realistic 2,000-row telco **customer-churn**
dataset with signal-bearing features plus deliberately seeded missing values,
outliers, and duplicate rows — so every part of the dashboard has something to show.

## License

MIT
