# Retail Inventory Management - Time Series Sales Forecasting

**Project**: AMS598 Time Series Forecasting  
**Objective**: Predict grocery store sales for the next 14 days using historical data, promotions, and economic indicators  
**Dataset**: Favorita Grocery Sales (Ecuador, 2017)

---

## 📊 Project Overview

This project implements a comprehensive time series forecasting solution for retail sales prediction across 54 stores and ~4,100 products in Ecuador. The goal is to forecast unit sales 14 days ahead for each store-item combination to optimize inventory management.

### Key Highlights
- **23.8M+ transactions** analyzed
- **221,400 unique time series** (store-item combinations)
- **70+ engineered features** including lag, rolling, and temporal features
- **14-day forecast horizon** matching business requirements

---

## 🗂️ Repository Structure

```
├── Final_code/
│   └── data_analysis.ipynb          # Complete analysis notebook
├── FEATURE_DICTIONARY.md            # Detailed feature descriptions & modeling guide
├── PROJECT_SUMMARY.md               # Comprehensive project documentation
├── FEATURE_ENGINEERING_README.md    # MPI4py parallelization guide
├── SLURM_INSTRUCTIONS.md            # HPC cluster deployment guide
├── README_SLURM_QUICK.md            # Quick start for SLURM
├── FILES_SUMMARY.md                 # Overview of all project files
└── .gitignore                       # Excludes large data files
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/aswithabukka/Reatil_inventory_management.git
cd Reatil_inventory_management
```

### 2. View the Analysis
Open `Final_code/data_analysis.ipynb` in Jupyter Notebook or JupyterLab to see:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Comprehensive feature engineering
- Train-test split strategy

### 3. Review Documentation
- **Start here**: `PROJECT_SUMMARY.md` - Complete project overview
- **For modeling**: `FEATURE_DICTIONARY.md` - All 71 features explained with keep/drop recommendations
- **For HPC deployment**: `SLURM_INSTRUCTIONS.md` - Run on computing clusters

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
- **Training Set**: ~23.3M rows (Jan 1 - Aug 1, 2017)
- **Test Set**: ~500K rows (Aug 2 - Aug 15, 2017)
- **40 recommended features** for modeling (from 71 total)

---

## 🎯 Recommended Features for Modeling

### Top 10 Most Important Features:
1. `sales_lag_7` - Last week same day (strongest predictor)
2. `sales_rolling_mean_7` - 7-day average trend
3. `store_item_dow_avg_sales` - Store-item day-of-week pattern
4. `sales_lag_1` - Yesterday's sales
5. `day_of_week` - Weekly seasonality
6. `onpromotion` - Current promotion status
7. `sales_lag_14` - 2 weeks ago sales
8. `transactions` - Store traffic indicator
9. `month` - Monthly seasonality
10. `store_item_month_avg_sales` - Monthly pattern

**See `FEATURE_DICTIONARY.md` for complete list of 40 recommended features.**

---

## 🔬 Methodology

### 1. Data Cleaning
- Removed duplicates
- Validated data types
- Outlier detection using IQR method

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
- **Train**: First 7 months (Jan-Aug 1)
- **Test**: Last 14 days (Aug 2-15) - matches forecast horizon

---

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib, seaborn** - Visualization
- **scikit-learn** - Preprocessing and encoding
- **Jupyter Notebook** - Interactive analysis

### Recommended for Modeling:
- **LightGBM** - Gradient boosting (fast, accurate for time series)
- **MPI4py** - Parallel processing for large-scale training

---

## 📊 Dataset Details

### Original Data
- **Source**: Favorita Grocery Sales (Ecuador)
- **Time Period**: January 1, 2017 - August 15, 2017
- **Rows**: 23,800,000+ transactions
- **Stores**: 54 stores across 22 cities in 16 states
- **Products**: ~4,100 unique items across 33 product families

### Key Variables
- **Target**: `unit_sales` (number of units sold)
- **Store Info**: Location (city, state), type (A-E), cluster
- **Product Info**: Family, class, perishability
- **Promotions**: Binary indicator for promotional items
- **Economic**: Daily oil prices (Ecuador economy indicator)
- **Calendar**: Holidays, events, workdays

---

## 📝 Documentation

| Document | Description |
|----------|-------------|
| `PROJECT_SUMMARY.md` | Complete project overview, methodology, and insights |
| `FEATURE_DICTIONARY.md` | Detailed description of all 71 features with modeling recommendations |
| `FEATURE_ENGINEERING_README.md` | Guide for MPI4py parallelization |
| `SLURM_INSTRUCTIONS.md` | Instructions for running on HPC clusters |
| `README_SLURM_QUICK.md` | Quick start guide for SLURM jobs |
| `FILES_SUMMARY.md` | Overview of all project files and usage |

---

## 🎓 Key Insights

### Business Insights
- **Promotions work**: Significant sales lift during promotional periods
- **Weekly patterns**: Strong day-of-week effects (weekends differ from weekdays)
- **Payday effects**: Sales spike on 15th and end of month
- **Store heterogeneity**: Different store types have distinct sales patterns
- **Product lifecycle**: New items sell differently than established products

### Technical Insights
- **Lag features are critical**: Historical sales are the strongest predictors
- **Weekly seasonality dominates**: `sales_lag_7` and `day_of_week` are top features
- **Store-item patterns matter**: Aggregated features capture local patterns
- **Data leakage prevention**: Proper use of `shift(1)` is essential
- **Missing values are expected**: First N days have no lag/rolling features (filled with 0)

---

## 🚀 Next Steps

### For Model Building:
1. Load processed datasets (`train_2017_ts_train.parquet`, `train_2017_ts_test.parquet`)
2. Select 40 recommended features from `FEATURE_DICTIONARY.md`
3. Train LightGBM model with hyperparameter tuning
4. Evaluate on test set (last 14 days)
5. Analyze feature importance
6. Generate 14-day forecasts

### For Production Deployment:
1. Implement MPI4py for parallel training (see `FEATURE_ENGINEERING_README.md`)
2. Deploy on HPC cluster using SLURM (see `SLURM_INSTRUCTIONS.md`)
3. Set up automated retraining pipeline
4. Create monitoring dashboard for forecast accuracy

---

## 📧 Contact

**Project**: AMS598 Time Series Forecasting  
**Repository**: https://github.com/aswithabukka/Reatil_inventory_management

---

## 📄 License

This project is for academic purposes (AMS598 course project).

---

## 🙏 Acknowledgments

- **Dataset**: Favorita Grocery Sales (Kaggle)
- **Course**: AMS598 - Time Series Analysis
- **Tools**: Python ecosystem (pandas, scikit-learn, matplotlib)

---

**Last Updated**: November 23, 2025
