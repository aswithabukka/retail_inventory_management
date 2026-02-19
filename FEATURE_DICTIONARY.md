# Feature Dictionary & Modeling Guide

**Project**: Time Series Sales Forecasting  
**Dataset**: Train 2017 with Engineered Features  
**Total Features**: 71 columns  
**Date**: November 23, 2025

---

## 📊 Complete Feature Summary

### 🔑 Identifier Columns (4 columns)

| Column | Description | Data Type | Use in Model |
|--------|-------------|-----------|--------------|
| `id` | Unique row identifier | Integer | ❌ Drop - Not predictive |
| `date` | Transaction date | Date | ❌ Drop - Use temporal features instead |
| `store_nbr` | Store identifier (1-54) | Integer | ⚠️ Use for grouping, not as feature |
| `item_nbr` | Product identifier | Integer | ⚠️ Use for grouping, not as feature |

**Note**: `store_nbr` and `item_nbr` are used for grouping time series but should not be used as direct features (too many unique values, causes overfitting).

---

### 🎯 Target Variable (2 columns)

| Column | Description | Data Type | Use in Model |
|--------|-------------|-----------|--------------|
| `unit_sales` | **TARGET VARIABLE** - Number of units sold | Float | ❌ This is what you're predicting! |
| `unit_sales_raw` | Backup of original unit_sales | Float | ❌ Drop - Duplicate of target |

**Important**: `unit_sales` is your target (y). Never include it in your feature matrix (X).

---

### 🏪 Store Features (8 columns)

| Column | Description | Data Type | Unique Values | Use in Model |
|--------|-------------|-----------|---------------|--------------|
| `city` | Store city name | String | 22 cities | ❌ Drop - Use encoded version |
| `state` | Store state name | String | 16 states | ❌ Drop - Use encoded version |
| `type` | Store type (A, B, C, D, E) | String | 5 types | ❌ Drop - Use encoded version |
| `cluster` | Store cluster group | Integer | 17 clusters | ✅ **Keep** |
| `city_encoded` | City encoded as numbers | Integer | 22 | ✅ **Keep** |
| `state_encoded` | State encoded as numbers | Integer | 16 | ✅ **Keep** |
| `type_encoded` | Store type encoded as numbers | Integer | 5 | ✅ **Keep** |
| `transactions` | Daily store transactions | Float | Continuous | ✅ **Keep** - Strong predictor |

**Scaled versions** (Drop these):
- `transactions_scaled` - Standardized transactions ❌
- `transactions_minmax` - Min-max normalized transactions ❌

**Why keep encoded versions?** Tree-based models (LightGBM, XGBoost) work better with integer encodings than text.

---

### 📦 Product Features (4 columns)

| Column | Description | Data Type | Unique Values | Use in Model |
|--------|-------------|-----------|---------------|--------------|
| `family` | Product family/category | String | 33 families | ❌ Drop - Use encoded version |
| `family_encoded` | Product family encoded as numbers | Integer | 33 | ✅ **Keep** |
| `class` | Product class number | Integer | ~312 classes | ✅ **Keep** |
| `perishable` | Is product perishable? | Integer (0/1) | 2 | ✅ **Keep** |

**Business Context**:
- `family`: Categories like GROCERY, BEVERAGES, PRODUCE, etc.
- `class`: Subcategory within family
- `perishable`: Important for inventory and sales patterns

---

### 💰 Promotion & Economic Features (4 columns)

| Column | Description | Data Type | Use in Model |
|--------|-------------|-----------|--------------|
| `onpromotion` | Item on promotion today? | Boolean | ✅ **Keep** - Very important! |
| `dcoilwtico` | Daily oil price (Ecuador economy indicator) | Float | ✅ **Keep** |
| `dcoilwtico_scaled` | Standardized oil price | Float | ❌ Drop - Redundant |
| `dcoilwtico_minmax` | Min-max normalized oil price | Float | ❌ Drop - Redundant |

**Why oil price matters**: Ecuador is oil-dependent; oil prices affect consumer spending.

---

### 📅 Calendar Features (3 columns)

| Column | Description | Data Type | Use in Model |
|--------|-------------|-----------|--------------|
| `is_holiday` | Is today a holiday? | Integer (0/1) | ✅ **Keep** |
| `is_event` | Is today a special event? | Integer (0/1) | ✅ **Keep** |
| `is_work_day` | Is today a makeup work day? | Integer (0/1) | ✅ **Keep** |

**Business Context**:
- Holidays increase sales (celebrations)
- Events affect shopping patterns
- Work days on weekends decrease sales

---

### 📆 Temporal Features (10 columns)

Extracted from the `date` column:

| Column | Description | Range | Use in Model |
|--------|-------------|-------|--------------|
| `year` | Year | 2017 | ⚠️ Drop - Only one year |
| `month` | Month | 1-12 | ✅ **Keep** - Strong seasonality |
| `day` | Day of month | 1-31 | ⚠️ Optional - Less important |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | 0-6 | ✅ **Keep** - Strong weekly patterns |
| `day_of_year` | Day of year | 1-365 | ⚠️ Drop - Month captures this |
| `week_of_year` | Week number | 1-52 | ⚠️ Drop - Month captures this |
| `quarter` | Quarter | 1-4 | ⚠️ Drop - Month captures this |
| `is_weekend` | Is weekend? | 0/1 | ✅ **Keep** - Important! |
| `is_month_start` | First day of month? | 0/1 | ✅ **Keep** - Payday effect |
| `is_month_end` | Last day of month? | 0/1 | ✅ **Keep** - Payday effect |

**Why payday matters**: Ecuador pays public sector wages on 15th and last day of month → sales spikes.

---

### ⏰ Lag Features (7 columns) 🌟 **MOST IMPORTANT**

Historical sales values shifted back in time:

| Column | Description | Lookback | Use in Model |
|--------|-------------|----------|--------------|
| `sales_lag_1` | Sales 1 day ago | 1 day | ✅ **Keep** - Very important! |
| `sales_lag_2` | Sales 2 days ago | 2 days | ✅ **Keep** |
| `sales_lag_3` | Sales 3 days ago | 3 days | ✅ **Keep** |
| `sales_lag_7` | Sales 7 days ago (last week) | 1 week | ✅ **Keep** - Very important! |
| `sales_lag_14` | Sales 14 days ago | 2 weeks | ✅ **Keep** |
| `sales_lag_21` | Sales 21 days ago | 3 weeks | ✅ **Keep** |
| `sales_lag_28` | Sales 28 days ago | 4 weeks | ✅ **Keep** |

**Why these are critical**: Past sales are the strongest predictor of future sales. Weekly patterns (lag_7) are especially important.

---

### 📊 Rolling Statistics - 7-Day Window (4 columns)

| Column | Description | Use in Model |
|--------|-------------|--------------|
| `sales_rolling_mean_7` | 7-day average sales | ✅ **Keep** - Very important! |
| `sales_rolling_std_7` | 7-day sales volatility (standard deviation) | ✅ **Keep** |
| `sales_rolling_min_7` | 7-day minimum sales | ⚠️ Drop - Less important |
| `sales_rolling_max_7` | 7-day maximum sales | ⚠️ Drop - Less important |

**Purpose**: Captures recent trend and volatility. Smooths out daily noise.

---

### 📊 Rolling Statistics - 14-Day Window (4 columns)

| Column | Description | Use in Model |
|--------|-------------|--------------|
| `sales_rolling_mean_14` | 14-day average sales | ✅ **Keep** - Important! |
| `sales_rolling_std_14` | 14-day sales volatility | ✅ **Keep** |
| `sales_rolling_min_14` | 14-day minimum sales | ⚠️ Drop - Redundant |
| `sales_rolling_max_14` | 14-day maximum sales | ⚠️ Drop - Redundant |

**Purpose**: Captures medium-term trends (2 weeks).

---

### 📊 Rolling Statistics - 30-Day Window (4 columns)

| Column | Description | Use in Model |
|--------|-------------|--------------|
| `sales_rolling_mean_30` | 30-day average sales | ✅ **Keep** - Important! |
| `sales_rolling_std_30` | 30-day sales volatility | ✅ **Keep** |
| `sales_rolling_min_30` | 30-day minimum sales | ❌ Drop - Redundant |
| `sales_rolling_max_30` | 30-day maximum sales | ❌ Drop - Redundant |

**Purpose**: Captures longer-term trends (1 month).

---

### 📈 EWMA Features (3 columns)

Exponentially Weighted Moving Average - gives more weight to recent observations:

| Column | Description | Use in Model |
|--------|-------------|--------------|
| `sales_ewm_7` | 7-day EWMA (recent trend) | ✅ **Keep** |
| `sales_ewm_14` | 14-day EWMA | ✅ **Keep** |
| `sales_ewm_30` | 30-day EWMA | ⚠️ Drop - Redundant with rolling_mean_30 |

**Difference from rolling mean**: EWMA gives exponentially more weight to recent days, better for detecting trend changes.

---

### 📉 Difference Features (4 columns)

Captures change/momentum in sales:

| Column | Description | Use in Model |
|--------|-------------|--------------|
| `sales_diff_1` | Day-over-day sales change (today - yesterday) | ✅ **Keep** |
| `sales_diff_7` | Week-over-week sales change (today - 7 days ago) | ✅ **Keep** |
| `sales_pct_change_1` | Day-over-day % change | ⚠️ Drop - Similar to diff_1 |
| `sales_pct_change_7` | Week-over-week % change | ⚠️ Drop - Similar to diff_7 |

**Purpose**: Detects acceleration/deceleration in sales. Positive = growing, negative = declining.

---

### 🎯 Store-Item Specific Features (2 columns)

Average sales patterns for each store-item combination:

| Column | Description | Use in Model |
|--------|-------------|--------------|
| `store_item_dow_avg_sales` | Average sales for this store-item on this day of week | ✅ **Keep** - Very important! |
| `store_item_month_avg_sales` | Average sales for this store-item in this month | ✅ **Keep** - Important! |

**Example**: Store #5, Item #100 typically sells 10 units on Mondays → `store_item_dow_avg_sales` = 10 on Mondays.

**Why important**: Captures store-item specific patterns (some items sell better in certain stores on certain days).

---

### 🏷️ Promotion History Features (3 columns)

Historical promotion information:

| Column | Description | Use in Model |
|--------|-------------|--------------|
| `promo_lag_7` | Was item on promotion 7 days ago? | ✅ **Keep** |
| `promo_lag_14` | Was item on promotion 14 days ago? | ✅ **Keep** |
| `promo_rolling_30` | Promotion rate in last 30 days (0-1) | ✅ **Keep** |

**Purpose**: Captures promotion effects and patterns. Items frequently on promotion have different baseline sales.

---

### 📈 Trend Features (2 columns)

| Column | Description | Use in Model |
|--------|-------------|--------------|
| `days_since_first_sale` | Days since item first appeared in store | ✅ **Keep** - Product lifecycle |
| `sales_momentum_7` | Current sales / 7-day average (ratio) | ✅ **Keep** - Trending indicator |

**Purpose**:
- `days_since_first_sale`: New products sell differently than established ones
- `sales_momentum_7`: Values > 1 = trending up, < 1 = trending down

---

## 🎯 Final Recommendations for Model Building

### ✅ **KEEP THESE FEATURES** (40 features)

```python
model_features = [
    # Store features (5)
    'cluster', 'city_encoded', 'state_encoded', 'type_encoded', 'transactions',
    
    # Product features (3)
    'family_encoded', 'class', 'perishable',
    
    # Promotion & economic (2)
    'onpromotion', 'dcoilwtico',
    
    # Calendar (3)
    'is_holiday', 'is_event', 'is_work_day',
    
    # Temporal (5)
    'month', 'day_of_week', 'is_weekend', 'is_month_start', 'is_month_end',
    
    # Lag features (7) - MOST IMPORTANT!
    'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_7',
    'sales_lag_14', 'sales_lag_21', 'sales_lag_28',
    
    # Rolling statistics (6)
    'sales_rolling_mean_7', 'sales_rolling_std_7',
    'sales_rolling_mean_14', 'sales_rolling_std_14',
    'sales_rolling_mean_30', 'sales_rolling_std_30',
    
    # EWMA (2)
    'sales_ewm_7', 'sales_ewm_14',
    
    # Difference (2)
    'sales_diff_1', 'sales_diff_7',
    
    # Store-item specific (2)
    'store_item_dow_avg_sales', 'store_item_month_avg_sales',
    
    # Promotion history (3)
    'promo_lag_7', 'promo_lag_14', 'promo_rolling_30',
    
    # Trend (2)
    'days_since_first_sale', 'sales_momentum_7'
]
```

---

### ❌ **DROP THESE FEATURES** (28 features)

```python
drop_cols = [
    # Identifiers (not features)
    'id', 'date',
    
    # Target variable duplicates
    'unit_sales_raw',
    
    # Text versions (use encoded instead)
    'city', 'state', 'type', 'family',
    
    # Scaled versions (redundant with original)
    'unit_sales_scaled', 'transactions_scaled', 'dcoilwtico_scaled',
    'unit_sales_minmax', 'transactions_minmax', 'dcoilwtico_minmax',
    
    # Less important temporal
    'year',  # Only 2017 data
    'day', 'day_of_year', 'week_of_year', 'quarter',
    
    # Redundant rolling features
    'sales_rolling_min_7', 'sales_rolling_max_7',
    'sales_rolling_min_14', 'sales_rolling_max_14',
    'sales_rolling_min_30', 'sales_rolling_max_30',
    
    # Redundant EWMA
    'sales_ewm_30',
    
    # Redundant difference features
    'sales_pct_change_1', 'sales_pct_change_7'
]
```

---

### ⚠️ **GROUPING COLUMNS** (Not features, but needed)

```python
grouping_cols = ['store_nbr', 'item_nbr']
```

**Important**: These identify the time series but should NOT be used as features in the model (too many unique values → overfitting).

---

### 🎯 **TARGET VARIABLE**

```python
target = 'unit_sales'
```

---

## 📋 Complete Model Preparation Code

```python
import pandas as pd
import numpy as np

# Load processed data
train_df = pd.read_parquet('train_2017_ts_train.parquet')
test_df = pd.read_parquet('train_2017_ts_test.parquet')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Define feature sets
grouping_cols = ['store_nbr', 'item_nbr']
target = 'unit_sales'

model_features = [
    # Store features
    'cluster', 'city_encoded', 'state_encoded', 'type_encoded', 'transactions',
    
    # Product features
    'family_encoded', 'class', 'perishable',
    
    # Promotion & economic
    'onpromotion', 'dcoilwtico',
    
    # Calendar
    'is_holiday', 'is_event', 'is_work_day',
    
    # Temporal
    'month', 'day_of_week', 'is_weekend', 'is_month_start', 'is_month_end',
    
    # Lag features
    'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_7',
    'sales_lag_14', 'sales_lag_21', 'sales_lag_28',
    
    # Rolling statistics
    'sales_rolling_mean_7', 'sales_rolling_std_7',
    'sales_rolling_mean_14', 'sales_rolling_std_14',
    'sales_rolling_mean_30', 'sales_rolling_std_30',
    
    # EWMA
    'sales_ewm_7', 'sales_ewm_14',
    
    # Difference
    'sales_diff_1', 'sales_diff_7',
    
    # Store-item specific
    'store_item_dow_avg_sales', 'store_item_month_avg_sales',
    
    # Promotion history
    'promo_lag_7', 'promo_lag_14', 'promo_rolling_30',
    
    # Trend
    'days_since_first_sale', 'sales_momentum_7'
]

# Create feature matrices
X_train = train_df[model_features]
y_train = train_df[target]

X_test = test_df[model_features]
y_test = test_df[target]

# Verify
print(f"\n{'='*60}")
print("MODEL PREPARATION SUMMARY")
print(f"{'='*60}")
print(f"Number of features: {len(model_features)}")
print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Target variable: {target}")
print(f"\nFeature matrix shape: {X_train.shape}")
print(f"Target vector shape: {y_train.shape}")
print(f"\nNo missing values in features: {X_train.isnull().sum().sum() == 0}")
print(f"No missing values in target: {y_train.isnull().sum() == 0}")
print(f"{'='*60}")
```

---

## 🌟 Expected Feature Importance (Top 15)

Based on time series forecasting best practices:

| Rank | Feature | Expected Importance | Why |
|------|---------|-------------------|-----|
| 1 | `sales_lag_7` | Very High | Last week same day |
| 2 | `sales_rolling_mean_7` | Very High | Recent trend |
| 3 | `store_item_dow_avg_sales` | Very High | Store-item day pattern |
| 4 | `sales_lag_1` | High | Yesterday's sales |
| 5 | `day_of_week` | High | Weekly seasonality |
| 6 | `onpromotion` | High | Current promotion |
| 7 | `sales_lag_14` | High | 2 weeks ago |
| 8 | `transactions` | High | Store traffic |
| 9 | `month` | Medium-High | Monthly seasonality |
| 10 | `store_item_month_avg_sales` | Medium-High | Monthly pattern |
| 11 | `sales_ewm_7` | Medium | Recent weighted trend |
| 12 | `sales_rolling_std_7` | Medium | Volatility |
| 13 | `family_encoded` | Medium | Product category |
| 14 | `is_weekend` | Medium | Weekend effect |
| 15 | `sales_momentum_7` | Medium | Trending indicator |

**Note**: Actual importance will vary by model and can be extracted after training using:
```python
# For LightGBM
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': model_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)
```

---

## 📊 Feature Categories Summary

| Category | Count | Keep | Drop |
|----------|-------|------|------|
| **Identifiers** | 4 | 0 | 4 |
| **Target** | 2 | 0 | 2 |
| **Store Features** | 8 | 5 | 3 |
| **Product Features** | 4 | 3 | 1 |
| **Promotion/Economic** | 4 | 2 | 2 |
| **Calendar** | 3 | 3 | 0 |
| **Temporal** | 10 | 5 | 5 |
| **Lag Features** | 7 | 7 | 0 |
| **Rolling Stats** | 12 | 6 | 6 |
| **EWMA** | 3 | 2 | 1 |
| **Difference** | 4 | 2 | 2 |
| **Store-Item Specific** | 2 | 2 | 0 |
| **Promotion History** | 3 | 3 | 0 |
| **Trend** | 2 | 2 | 0 |
| **Scaled Features** | 6 | 0 | 6 |
| **TOTAL** | **71** | **40** | **31** |

---

## 🚀 Next Steps

1. **Load processed datasets**:
   - `train_2017_ts_train.parquet`
   - `train_2017_ts_test.parquet`

2. **Select features** using the `model_features` list above

3. **Train model** (LightGBM recommended):
   ```python
   import lightgbm as lgb
   
   model = lgb.LGBMRegressor(
       n_estimators=1000,
       learning_rate=0.05,
       max_depth=8,
       num_leaves=31,
       random_state=42
   )
   
   model.fit(X_train, y_train)
   ```

4. **Evaluate** on test set (last 14 days)

5. **Parallelize with MPI4py** for faster training across store-item combinations

---

## 📝 Notes

- **Data Leakage Prevention**: All lag and rolling features use `shift(1)` to prevent using future information
- **Missing Values**: Filled with 0 (represents "no historical data available")
- **Scaling**: Not needed for tree-based models (LightGBM, XGBoost, Random Forest)
- **Encoding**: Label encoding used (sufficient for tree-based models)
- **Time Series Split**: Last 14 days as test set (matches forecast horizon)

---

**Document Version**: 1.0  
**Last Updated**: November 23, 2025  
**Author**: Data Analysis Pipeline  
**Contact**: See PROJECT_SUMMARY.md for full project details
