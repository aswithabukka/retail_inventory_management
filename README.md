# Retail Inventory Management - Time Series Sales Forecasting

**Project**: Retail Inventory Management - Time Series Forecasting
**Objective**: Predict grocery store sales for the next 14 days using historical data, promotions, and economic indicators
**Dataset**: Favorita Grocery Sales (Ecuador, 2017)

---

## 📊 Project Overview

This project implements a comprehensive time series forecasting solution for retail sales prediction across 54 stores and ~4,100 products in Ecuador. The goal is to forecast unit sales 14 days ahead for each store-item combination to optimize inventory management.

### Key Highlights
- **23.8M+ transactions** analyzed
- **221,400 unique time series** (store-item combinations)
- **72 engineered features** including lag, rolling, and temporal features
- **14-day forecast horizon** matching business requirements
- **Random Forest model** trained on 22.3M rows using PySpark on Google Cloud

---

## 🗂️ Repository Structure

```
├── data_analysis.ipynb                  # Data cleaning, EDA, and feature engineering notebook
├── Pyspark_datapreprocessing.ipynb      # PySpark-based data preprocessing notebook
├── inventory_demamd_prediction.ipynb      # PySpark Random Forest model - training & evaluation
├── FEATURE_DICTIONARY.md               # Detailed feature descriptions & modeling guide
├── PROJECT_SUMMARY.md                  # Comprehensive project documentation
└── README.md                           # Project overview (this file)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/aswithabukka/retail_inventory_management.git
cd retail_inventory_management
```

### 2. View the Analysis
Open `data_analysis.ipynb` in Jupyter Notebook or JupyterLab to see:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Comprehensive feature engineering
- Train-test split strategy

### 3. View the Model
Open `inventory_demamd_prediction.ipynb` to see:
- PySpark Random Forest model training on full 22M+ row dataset
- Feature importance analysis
- Model evaluation with RMSE metrics
- Actual vs predicted sales results

### 4. Review Documentation
- **Start here**: `PROJECT_SUMMARY.md` - Complete project overview
- **For modeling**: `FEATURE_DICTIONARY.md` - All 72 features explained with keep/drop recommendations

---

## 📈 Key Features

### Data Processing
✅ **Data Cleaning**: Handled missing values, duplicates, and outliers
✅ **Categorical Encoding**: Label encoding for cities, states, store types, product families
✅ **Temporal Features**: Extracted year, month, day of week, holidays, weekends

### Advanced Feature Engineering
✅ **Lag Features** (7 features): Historical sales (1, 2, 3, 7, 14, 21, 28 days ago)
✅ **Rolling Statistics** (12 features): 7, 14, 30-day windows (mean, std, min, max)
✅ **EWMA Features** (3 features): Exponentially weighted moving averages
✅ **Difference Features** (4 features): Day-over-day and week-over-week changes
✅ **Store-Item Patterns** (2 features): Average sales by day of week and month
✅ **Promotion History** (3 features): Lagged and rolling promotion indicators
✅ **Trend Features** (2 features): Product lifecycle and momentum indicators

### Model-Ready Outputs
- **Training Set**: ~22.3M rows (Jan 1 - Aug 1, 2017)
- **Test Set**: ~1.46M rows (Aug 2 - Aug 15, 2017)
- **44 features** used for modeling (from 72 total)

---

## 🤖 Model: PySpark Random Forest Regressor

### Infrastructure
- **Framework**: Apache Spark MLlib (PySpark)
- **Cloud Platform**: Google Cloud Platform (GCP)
- **Data Storage**: Google Cloud Storage (`gs://inv-demand-bucket/`)
- **Model Storage**: `gs://inv-demand-bucket/models`

### Configuration
| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| Number of Trees | 50 |
| Max Depth | 10 |
| Random Seed | 42 |
| Partitions | 16 |
| Features Used | 44 |
| Training Rows | 22,346,680 |

### Pipeline
```
Data Loading (GCS Parquet)
    → Data Sanitization (Infinity/NaN/Null handling, Boolean → Integer)
    → VectorAssembler (44 features → feature vector)
    → RandomForestRegressor
    → PipelineModel (saved to GCS)
```

### Model Performance
| Metric | Value |
|--------|-------|
| **RMSE** (Root Mean Squared Error) | **13.6885** |
| Test Set Size | 1,461,581 rows |
| Test Period | Aug 2 - Aug 15, 2017 (14 days) |

### Top 10 Feature Importances (Actual Results)

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | `store_item_month_avg_sales` | 0.1490 | Monthly average sales per store-item |
| 2 | `sales_momentum_7` | 0.1088 | Current sales vs 7-day rolling average |
| 3 | `store_item_dow_avg_sales` | 0.0876 | Day-of-week average per store-item |
| 4 | `sales_rolling_mean_7` | 0.0834 | 7-day rolling average sales |
| 5 | `sales_diff_1` | 0.0823 | Day-over-day sales change |
| 6 | `sales_ewm_14` | 0.0577 | 14-day exponentially weighted mean |
| 7 | `sales_rolling_mean_14` | 0.0470 | 14-day rolling average sales |
| 8 | `sales_diff_7` | 0.0357 | Week-over-week sales change |
| 9 | `sales_lag_1` | 0.0302 | Yesterday's sales |
| 10 | `transactions` | 0.0220 | Store daily transaction count |

### Sample Predictions
```
+---------+---------+--------+--------+------------------+-------+
|       id|store_nbr|item_nbr| Actual |         Predicted|   Diff|
+---------+---------+--------+--------+------------------+-------+
|124792795|        8| 1366212|     8.0|  7.486722209483122|   0.51|
|124046636|        5| 1999114|     1.0| 1.2147200753698628|  -0.21|
|124585499|        7| 1346628|     5.0|  4.582576117662974|   0.42|
|124687725|        7|  108797|     8.0|  8.024190262625309|  -0.02|
|124362990|        6| 2043849|     9.0|   8.8408151944928|   0.16|
+---------+---------+--------+--------+------------------+-------+
```

---

## 🎯 Recommended Features for Modeling

### Top 10 Most Important Features (from RF model):
1. `store_item_month_avg_sales` - Monthly store-item sales pattern (strongest predictor)
2. `sales_momentum_7` - Recent sales momentum vs rolling average
3. `store_item_dow_avg_sales` - Store-item day-of-week pattern
4. `sales_rolling_mean_7` - 7-day average trend
5. `sales_diff_1` - Day-over-day sales change
6. `sales_ewm_14` - 14-day exponentially weighted trend
7. `sales_rolling_mean_14` - 14-day average trend
8. `sales_diff_7` - Week-over-week change
9. `sales_lag_1` - Yesterday's sales
10. `transactions` - Store traffic indicator

**See `FEATURE_DICTIONARY.md` for complete list of all 72 features.**

---

## 🔬 Methodology

### 1. Data Cleaning
- Removed duplicates
- Validated data types
- Outlier detection using IQR method
- Fixed impossible negatives in sales and transaction counts

### 2. Exploratory Data Analysis
- **Promotion Impact**: 30-50% sales increase during promotions
- **Weekly Patterns**: Strong day-of-week seasonality
- **Monthly Patterns**: Payday effects (15th and end of month)
- **Store Types**: Type A stores have highest volume

### 3. Feature Engineering
- **Time Series Features**: Lag, rolling, EWMA, difference features
- **Data Leakage Prevention**: All features use `shift(1)` to avoid future information
- **Missing Value Handling**: Filled with 0 (represents no historical data)

### 4. Train-Test Split
- **Strategy**: Time-based split (preserves temporal order)
- **Train**: First 7 months (Jan 1 - Aug 1, 2017) → 22,346,680 rows
- **Test**: Last 14 days (Aug 2-15, 2017) → 1,461,581 rows

### 5. Model Training (PySpark on GCP)
- Loaded Parquet data from GCS
- Applied feature sanitization (Infinity/NaN → 0, Boolean → Integer)
- Built Spark ML Pipeline: VectorAssembler + RandomForestRegressor
- Trained on full 22M row dataset across 16 partitions
- Saved trained PipelineModel back to GCS

---

## 🛠️ Technologies Used

### Data Processing & Analysis
- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib, seaborn** - Visualization
- **scikit-learn** - Preprocessing and encoding
- **Jupyter Notebook** - Interactive analysis

### Distributed Model Training
- **Apache Spark (PySpark)** - Distributed ML training on 22M+ rows
- **PySpark MLlib** - Random Forest Regressor, VectorAssembler, Pipeline
- **Google Cloud Platform (GCP)** - Cloud compute infrastructure
- **Google Cloud Storage (GCS)** - Data and model storage

---

## 📊 Dataset Details

### Original Data
- **Source**: Favorita Grocery Sales (Ecuador)
- **Time Period**: January 1, 2017 - August 15, 2017
- **Rows**: 23,808,261 transactions
- **Stores**: 54 stores across 22 cities in 16 states
- **Products**: ~4,100 unique items across 33 product families

### Key Variables
- **Target**: `unit_sales` (number of units sold)
- **Store Info**: Location (city, state), type (A-E), cluster
- **Product Info**: Family, class, perishability
- **Promotions**: Binary indicator for promotional items
- **Economic**: Daily oil prices (Ecuador economy indicator)
- **Calendar**: Holidays, events, workdays

### Processed Datasets (on GCS)
| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `train_2017_ts_train.parquet` | 22,346,680 | 72 | Training set with all features |
| `train_2017_ts_test.parquet` | 1,461,581 | 72 | Test set (last 14 days) |
| `train_2017_ts_features_full.parquet` | 23,808,261 | 72 | Full dataset with features |

---

## 📝 Documentation

| Document | Description |
|----------|-------------|
| `data_analysis.ipynb` | Complete data cleaning, EDA, and feature engineering notebook |
| `Pyspark_datapreprocessing.ipynb` | PySpark-based data preprocessing pipeline |
| `inventory_demamd_prediction.ipynb` | PySpark Random Forest model training, evaluation, and feature importance |
| `PROJECT_SUMMARY.md` | Complete project overview, methodology, and insights |
| `FEATURE_DICTIONARY.md` | Detailed description of all 72 features with modeling recommendations |

---

## 🎓 Key Insights

### Business Insights
- **Store-item patterns dominate**: Monthly and day-of-week averages per store-item are the strongest predictors
- **Momentum matters**: Recent sales momentum vs rolling average is the second most important feature
- **Promotions work**: Significant sales lift during promotional periods
- **Weekly patterns**: Strong day-of-week effects (weekends differ from weekdays)
- **Payday effects**: Sales spike on 15th and end of month
- **Store heterogeneity**: Different store types have distinct sales patterns

### Technical Insights
- **Store-item aggregates are critical**: `store_item_month_avg_sales` (0.149) outperforms raw lag features
- **Momentum over raw lags**: `sales_momentum_7` (0.109) ranks higher than `sales_lag_1` (0.030)
- **Difference features are valuable**: `sales_diff_1` (0.082) captures day-over-day change effectively
- **PySpark scales well**: Full 22M row training completed successfully on GCP with 16 partitions
- **Data leakage prevention**: Proper use of `shift(1)` is essential for valid evaluation

## 🙏 Acknowledgments

- **Dataset**: Favorita Grocery Sales (Kaggle)
- **Tools**: Python ecosystem (pandas, scikit-learn, matplotlib), Apache Spark, Google Cloud Platform

