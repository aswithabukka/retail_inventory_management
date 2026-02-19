# Time Series Forecasting Project - Data Preparation Summary

**Project Goal**: Predict sales for the next 14 days for specific store-item combinations

**Dataset**: `train_2017_clean.parquet` (23.8 million rows, 20 columns)

**Date Range**: January 1, 2017 to August 15, 2017

---

## 📊 Dataset Overview

### Original Dataset Statistics
- **Total Records**: 23,808,261 rows
- **Unique Stores**: 54 stores
- **Unique Items**: ~4,100 items  
- **Store-Item Combinations**: ~221,400 unique pairs
- **Target Variable**: `unit_sales` (continuous variable representing number of units sold)
- **Granularity**: Daily sales data per store-item combination

### Key Features in Original Dataset
| Feature | Type | Description |
|---------|------|-------------|
| `id` | int64 | Unique transaction identifier |
| `date` | datetime64 | Transaction date |
| `store_nbr` | int64 | Store identifier (1-54) |
| `item_nbr` | int64 | Product identifier |
| `unit_sales` | float64 | **TARGET VARIABLE** - Units sold |
| `onpromotion` | object/bool | Whether item was on promotion |
| `city` | category | Store city location |
| `state` | category | Store state location |
| `type` | category | Store type (A, B, C, D, E) |
| `cluster` | int64 | Store cluster grouping |
| `family` | category | Product family/category |
| `class` | int64 | Product class |
| `perishable` | int64 | Whether product is perishable (0/1) |
| `transactions` | int64 | Number of transactions at store |
| `dcoilwtico` | float64 | Daily oil price (economic indicator) |
| `is_holiday` | int64 | Holiday indicator (0/1) |
| `is_event` | int64 | Special event indicator (0/1) |
| `is_work_day` | int64 | Workday indicator (0/1) |
| `unit_sales_raw` | float64 | Raw sales before processing |

**Note**: `log_sales` feature was removed as `unit_sales` is the actual target variable.

---

## 🧹 1. Data Cleaning Performed

### Missing Values
- ✅ **Result**: No missing values found in the dataset
- All 20 columns had complete data

### Duplicates
- ✅ **Result**: No duplicate rows found
- Dataset integrity verified

### Outliers
- **Method**: IQR (Interquartile Range) method applied
- **Finding**: Some outliers detected in sales data (expected for retail)
- **Action**: Kept outliers as they represent legitimate high-sales events (holidays, promotions)

### Data Type Validation
- Converted `onpromotion` from object to boolean where needed
- Ensured `date` column is datetime64 format
- Verified categorical columns are properly typed

---

## 🔧 2. Data Preprocessing Performed

### Categorical Encoding
- **Label Encoding** applied to:
  - `city` (22 unique cities)
  - `state` (16 unique states)
  - `type` (5 store types: A, B, C, D, E)
  - `family` (33 product families)

### Numerical Scaling
Two types of scaling applied for modeling flexibility:
1. **StandardScaler (Z-score normalization)**:
   - `unit_sales_scaled`
   - `transactions_scaled`
   - `dcoilwtico_scaled`

2. **MinMaxScaler (0-1 normalization)**:
   - `unit_sales_minmax`
   - `transactions_minmax`
   - `dcoilwtico_minmax`

---

## 📈 3. Exploratory Data Analysis (EDA) Insights

### Target Variable (`unit_sales`)
- **Distribution**: Right-skewed (typical for sales data)
- **Mean**: ~7.5 units
- **Median**: ~3 units
- **Range**: 0 to several hundred units

### Key Business Insights

#### Promotion Impact
- **Finding**: Promotions significantly increase average sales
- Items on promotion show 30-50% higher sales on average

#### Store Type Performance
- **Type A stores**: Highest total sales volume
- **Type D stores**: Highest average sales per transaction
- Store type is a strong predictor of sales

#### Product Family Trends
- **Top sellers**: GROCERY I, BEVERAGES, PRODUCE
- **Seasonal patterns**: Strong variations across product families
- Perishable items show different patterns than non-perishable

#### Temporal Patterns
- **Weekly seasonality**: Clear weekend vs weekday patterns
- **Monthly trends**: End-of-month spikes (payday effect)
- **Holiday impact**: Significant sales increases during holidays
- **Oil price correlation**: Weak negative correlation with sales

#### Day of Week Effects
- **Weekends**: Generally higher sales
- **Mondays**: Typically lowest sales day
- Pattern varies by product family

---

## 🎯 4. Feature Engineering for Time Series Forecasting

### Critical Design Principle
**NO DATA LEAKAGE**: All features use `shift(1)` or proper lagging to ensure we only use past information to predict future sales.

### Feature Categories Created

#### A. Temporal Features (11 features)
Capture calendar-based patterns:
- `year`, `month`, `day`, `day_of_week`
- `day_of_year`, `week_of_year`, `quarter`
- `is_weekend`, `is_month_start`, `is_month_end`

**Why useful**: Captures seasonality, weekly patterns, and calendar effects

#### B. Lag Features (4 features)
Historical sales at specific time points:
- `sales_lag_1`: Yesterday's sales
- `sales_lag_7`: Sales 7 days ago (same day last week)
- `sales_lag_14`: Sales 14 days ago (2 weeks ago)
- `sales_lag_28`: Sales 28 days ago (4 weeks ago)

**Why useful**: 
- Captures auto-correlation (sales tend to be similar to recent past)
- `lag_7` captures weekly seasonality
- `lag_28` captures monthly patterns
- **Most important features for time series forecasting**

#### C. Rolling Window Features (6 features)
Moving statistics over time windows:

**7-Day Window**:
- `sales_rolling_mean_7`: 7-day average sales
- `sales_rolling_std_7`: 7-day sales volatility

**14-Day Window**:
- `sales_rolling_mean_14`: 14-day average sales
- `sales_rolling_std_14`: 14-day sales volatility

**30-Day Window**:
- `sales_rolling_mean_30`: 30-day average sales
- `sales_rolling_std_30`: 30-day sales volatility

**Why useful**:
- Smooths out noise and captures trends
- `mean` shows average performance level
- `std` shows stability/volatility of sales
- Different windows capture short/medium/long-term patterns

#### D. Exponentially Weighted Moving Average (2 features)
Recent observations weighted more heavily:
- `sales_ewm_7`: 7-day EWMA
- `sales_ewm_14`: 14-day EWMA

**Why useful**:
- Better captures recent trends than simple moving average
- Reacts faster to changes in sales patterns
- Gives more weight to recent data (more relevant for prediction)

#### E. Difference Features (2 features)
Rate of change:
- `sales_diff_1`: Day-over-day change
- `sales_diff_7`: Week-over-week change

**Why useful**:
- Captures momentum and growth rates
- Helps identify trending vs stable items
- Can improve stationarity for some models

### Total Features Created
- **Original features**: 19 (after removing log_sales)
- **New time series features**: ~25
- **Total features**: ~44

---

## 📦 5. Train-Test Split Strategy

### Time-Based Split (NO RANDOM SPLIT!)
**Critical for time series**: Must preserve temporal order

- **Training Set**: All data from start through August 1, 2017
  - Shape: ~23.3 million rows
  - Date range: 2017-01-01 to 2017-08-01

- **Test Set**: Last 14 days of data
  - Shape: ~500,000 rows  
  - Date range: 2017-08-02 to 2017-08-15
  - **Matches forecast horizon**: 14 days

### Why This Split?
1. **No data leakage**: Test set is strictly after training set
2. **Realistic evaluation**: Mimics real-world forecasting scenario
3. **Proper validation**: Tests model's ability to predict 14 days ahead

---

## 💾 6. Output Files Generated

### Processed Datasets
All saved in parquet format for efficient storage and loading:

1. **`train_2017_ts_train.parquet`**
   - Training dataset with all engineered features
   - Ready for model training
   - ~23.3M rows × 44 columns

2. **`train_2017_ts_test.parquet`**
   - Test dataset with all engineered features
   - For model evaluation
   - ~500K rows × 44 columns

### Notebooks
1. **`data_analysis.ipynb`** (Original - comprehensive but memory-intensive)
2. **`data_analysis_optimized.ipynb`** (Optimized for large datasets)

---

## 🚀 7. Next Steps for Your Team: Model Building with MPI4py

### Problem Statement
**Forecast unit_sales for the next 14 days for each store-item combination**

### Parallelization Strategy with MPI4py

#### Why MPI4py?
- **~221,400 store-item combinations** to forecast
- Each combination can be modeled independently
- Perfect for parallel processing across multiple cores/nodes

#### Recommended Approach

```python
from mpi4py import MPI
import pandas as pd

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load data (each process loads its own copy)
train_df = pd.read_parquet('train_2017_ts_train.parquet')
test_df = pd.read_parquet('train_2017_ts_test.parquet')

# Get all unique store-item combinations
store_item_pairs = train_df[['store_nbr', 'item_nbr']].drop_duplicates()
total_pairs = len(store_item_pairs)

# Split work across processes
pairs_per_process = total_pairs // size
start_idx = rank * pairs_per_process
end_idx = start_idx + pairs_per_process if rank < size - 1 else total_pairs

my_pairs = store_item_pairs.iloc[start_idx:end_idx]

print(f"Process {rank}: Handling {len(my_pairs)} store-item pairs")

# Train models for assigned pairs
results = []
for idx, (store, item) in my_pairs.iterrows():
    # Filter data for this store-item
    train_data = train_df[(train_df['store_nbr'] == store) & 
                          (train_df['item_nbr'] == item)]
    
    # Train model (your choice of algorithm)
    model = train_model(train_data)
    
    # Make 14-day forecast
    predictions = model.predict(14)
    
    # Store results
    results.append({
        'store_nbr': store,
        'item_nbr': item,
        'predictions': predictions
    })

# Gather results from all processes
all_results = comm.gather(results, root=0)

# Process 0 combines and saves final predictions
if rank == 0:
    final_predictions = combine_results(all_results)
    save_predictions(final_predictions)
```

### Model Selection Options

#### Option 1: Statistical Models (Fast, Interpretable)
- **ARIMA/SARIMA**: Classical time series
  - Pros: Well-understood, captures seasonality
  - Cons: Slow for many series, requires stationarity
  
- **Prophet** (Facebook):
  - Pros: Handles seasonality well, robust to missing data
  - Cons: May not capture complex interactions

#### Option 2: Machine Learning Models (Recommended)
- **XGBoost/LightGBM**:
  - Pros: Fast, handles many features, excellent performance
  - Cons: Requires feature engineering (already done!)
  - **Best choice for this dataset**

- **Random Forest**:
  - Pros: Robust, handles non-linearity
  - Cons: Slower than gradient boosting

#### Option 3: Deep Learning (For Complex Patterns)
- **LSTM/GRU**:
  - Pros: Captures long-term dependencies
  - Cons: Requires more data, slower training, harder to tune

- **Temporal Convolutional Networks (TCN)**:
  - Pros: Faster than LSTM, good for long sequences
  - Cons: More complex to implement

### Recommended Model: LightGBM

**Why LightGBM?**
1. ✅ Extremely fast training
2. ✅ Handles large datasets efficiently
3. ✅ Works well with engineered features
4. ✅ Built-in handling of categorical features
5. ✅ Excellent performance on tabular data
6. ✅ Easy to parallelize with MPI4py

**Example LightGBM Implementation**:
```python
import lightgbm as lgb

def train_model(train_data):
    # Define features (exclude target and identifiers)
    feature_cols = [col for col in train_data.columns 
                   if col not in ['unit_sales', 'date', 'id', 'store_nbr', 'item_nbr']]
    
    X = train_data[feature_cols]
    y = train_data['unit_sales']
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Train model
    train_set = lgb.Dataset(X, y)
    model = lgb.train(params, train_set, num_boost_round=100)
    
    return model
```

---

## 📊 8. Evaluation Metrics

### Recommended Metrics for Sales Forecasting

1. **RMSE (Root Mean Squared Error)**
   ```python
   rmse = np.sqrt(mean_squared_error(y_true, y_pred))
   ```
   - Penalizes large errors heavily
   - Same units as target variable

2. **MAE (Mean Absolute Error)**
   ```python
   mae = mean_absolute_error(y_true, y_pred)
   ```
   - More robust to outliers than RMSE
   - Easy to interpret

3. **MAPE (Mean Absolute Percentage Error)**
   ```python
   mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
   ```
   - Scale-independent
   - Good for comparing across different items

4. **SMAPE (Symmetric MAPE)**
   ```python
   smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
   ```
   - Bounded between 0-100%
   - Symmetric treatment of over/under-prediction

### Evaluation Strategy
```python
# Per store-item evaluation
for store, item in store_item_pairs:
    y_true = test_df[(test_df['store_nbr'] == store) & 
                     (test_df['item_nbr'] == item)]['unit_sales']
    y_pred = predictions[(store, item)]
    
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    
    # Store metrics

# Aggregate metrics
overall_rmse = np.mean(all_rmse)
overall_mae = np.mean(all_mae)
```

---

## 9. Feature Importance Analysis

After training, analyze which features are most important:

```python
# For LightGBM
importance = model.feature_importance()
feature_names = model.feature_name()

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print(importance_df.head(20))
```

**Expected Important Features**:
1. `sales_lag_7` (last week's sales)
2. `sales_rolling_mean_7` (recent average)
3. `sales_lag_1` (yesterday's sales)
4. `day_of_week` (weekly seasonality)
5. `onpromotion` (promotion effect)
6. `sales_ewm_7` (recent trend)
7. `transactions` (store activity)
8. `is_holiday` (special days)

---

##  10. Tips for Successful Implementation

### Memory Management
- Load data in chunks if needed
- Use `del` and `gc.collect()` to free memory
- Consider using `dask` for very large datasets

### Debugging Strategy
1. **Start small**: Test on 1 store-item pair first
2. **Scale gradually**: 10 pairs → 100 pairs → all pairs
3. **Monitor resources**: Check CPU, memory usage
4. **Log progress**: Print status updates from each MPI process

### Common Pitfalls to Avoid
❌ **Random train-test split** → Use time-based split  
❌ **Using future data** → Always use shift/lag  
❌ **Ignoring seasonality** → Include day_of_week, month features  
❌ **One model for all** → Train separate models per store-item  
❌ **Not handling zeros** → Many items have zero sales on some days  

### Performance Optimization
- Use `categorical_feature` parameter in LightGBM
- Enable early stopping to prevent overfitting
- Use GPU acceleration if available
- Cache intermediate results

---

## 📋 11. Deliverables Checklist

- [x] Data cleaned and validated
- [x] Time series features engineered
- [x] Train-test split created (time-based)
- [x] Processed datasets saved
- [ ] Model selection and implementation
- [ ] MPI4py parallelization setup
- [ ] Model training across all store-item pairs
- [ ] Predictions generated for 14-day horizon
- [ ] Evaluation metrics calculated
- [ ] Feature importance analysis
- [ ] Final results visualization
- [ ] Model deployment strategy

---

## 🔗 12. File Structure

```
AMS598_project/
├── train_2017_clean.parquet          # Original dataset
├── train_2017_ts_train.parquet       # Training set (ready for modeling)
├── train_2017_ts_test.parquet        # Test set (for evaluation)
├── data_analysis.ipynb               # Full analysis notebook
├── data_analysis_optimized.ipynb     # Memory-optimized version
├── PROJECT_SUMMARY.md                # This document
└── (future files)
    ├── train_model_mpi.py            # MPI4py training script
    ├── predictions.parquet           # Model predictions
    └── evaluation_results.csv        # Performance metrics
```



## 🎓 14. Key Takeaways

### What We Accomplished
✅ Cleaned and validated 23.8M rows of sales data  
✅ Created 25+ time series features optimized for forecasting  
✅ Properly split data to prevent leakage  
✅ Prepared data for parallel processing  
✅ Documented entire process for reproducibility  

### What Makes This Dataset Ready for Modeling
1. **No data leakage**: All features use only past information
2. **Proper time series features**: Lag, rolling, EWMA features
3. **Temporal ordering preserved**: Sorted by store, item, date
4. **Clean train-test split**: Last 14 days held out
5. **Scalable format**: Parquet files for fast I/O

### Success Metrics
Your model should aim for:
- **RMSE < 5 units** (good performance)
- **MAPE < 30%** (acceptable for retail)
- **Training time < 2 hours** (with MPI parallelization)

