# Exploratory Data Analysis & Integrity Report (Week 1)

## 1. Summary Statistics
- **Total Model Rows**: 6,627
- **Date Range**: 2012-03-29 to 2025-09-26
- **Unique Tickers**: 2 (RELIANCE.NS, TCS.NS...)
- **Target Label Balance (Overall)**: 42.09% positive

## 2. Annual Class Balance
| Year | Row Count | Positive Ratio (%) |
|---|---|---|
| 2012 | 366 | 46.17% |
| 2013 | 494 | 42.91% |
| 2014 | 480 | 40.21% |
| 2015 | 488 | 42.62% |
| 2016 | 488 | 45.08% |
| 2017 | 496 | 44.96% |
| 2018 | 490 | 45.51% |
| 2019 | 482 | 39.83% |
| 2020 | 500 | 41.80% |
| 2021 | 496 | 43.15% |
| 2022 | 496 | 38.71% |
| 2023 | 490 | 40.20% |
| 2024 | 492 | 42.89% |
| 2025 | 369 | 34.15% |

## 3. Feature Matrix Characteristics
- Features compute rolling statistics strictly backward in time.
- Macro data aligned via point-in-time NSE trading calendar.
- No forward return leaks into feature columns.
