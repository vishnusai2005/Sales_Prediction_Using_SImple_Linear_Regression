# 📈 Sales Prediction using Simple Linear Regression


## 📌 Problem Statement

> **Build a machine learning model that predicts Sales (in thousands of units) based on the money spent (in thousands of dollars) on different advertising platforms — TV, Radio, and Newspaper.**

The model focuses specifically on **TV advertising spend** as the primary predictor variable using **Simple Linear Regression (SLR)**.

---

## 📂 Project Structure

```
SALES_PREDICTION_SLR/
│
├── Advertising.csv              # Dataset with TV, Radio, Newspaper spend & Sales
├── Sales_prediction.ipynb       # Jupyter Notebook with complete ML pipeline
└── README.md                    # Project documentation (this file)
```

---

## 📊 Dataset Overview

**File:** `Advertising.csv`  
**Source:** Classic Advertising Dataset  
**Shape:** 200 rows × 4 feature columns  

| Column      | Type    | Description                                      |
|-------------|---------|--------------------------------------------------|
| `TV`        | float64 | Advertising budget spent on TV (in $thousands)   |
| `Radio`     | float64 | Advertising budget spent on Radio (in $thousands)|
| `Newspaper` | float64 | Advertising budget spent on Newspaper ($thousands)|
| `Sales`     | float64 | Sales generated (in thousands of units) — **Target** |

**Sample Data:**

| TV    | Radio | Newspaper | Sales |
|-------|-------|-----------|-------|
| 230.1 | 37.8  | 69.2      | 22.1  |
| 44.5  | 39.3  | 45.1      | 10.4  |
| 17.2  | 45.9  | 69.3      | 9.3   |

---

## 🛠️ Tech Stack & Libraries

| Library         | Purpose                                              |
|-----------------|------------------------------------------------------|
| `numpy`         | Numerical computations, array operations             |
| `pandas`        | Data loading, manipulation, and preprocessing        |
| `matplotlib`    | Plotting regression lines, scatter plots             |
| `seaborn`       | Statistical visualizations (boxplots, heatmaps, pairplots) |
| `statsmodels`   | OLS regression model building and statistical summary|
| `scikit-learn`  | Train-test split, MSE metric evaluation              |
| `warnings`      | Suppressing deprecation/runtime warnings             |

---

## 🔄 ML Pipeline Workflow

```
Load Dataset → Data Cleaning → Outlier Analysis → EDA
       → Feature Selection → Train-Test Split
              → OLS Model Building → Model Evaluation
                     → Predictions on Test Set → RMSE
```

---

## 🧪 Step-by-Step Explanation of Every Technique

---

### 1. 📥 Data Loading

```python
advertising = pd.read_csv("Advertising.csv", index_col=0)
advertising.head()
```

**What it does:**  
Loads the CSV file into a Pandas DataFrame. `index_col=0` sets the first unnamed index column as the DataFrame row index, preventing a spurious `Unnamed: 0` column from appearing in the data.

`.head()` displays the first 5 rows to quickly verify the data was loaded correctly.

---

### 2. 📋 Descriptive Statistics

```python
advertising.describe()
```

**What it does:**  
Generates an 8-number statistical summary for every numeric column:
- **count** — number of non-null values
- **mean** — arithmetic average
- **std** — standard deviation (spread of data)
- **min / max** — range boundaries
- **25%, 50%, 75%** — interquartile range (IQR quartiles)

**Why it matters:** Gives a first-pass understanding of data distribution, scale, and potential anomalies before any deeper analysis.

---

### 3. 🧹 Data Cleaning — Null Value Detection

```python
advertising.isnull().sum()
```

**What it does:**  
`isnull()` returns a boolean DataFrame where `True` marks missing values. `.sum()` tallies them column-wise.

**Why it matters:** Missing values (NaN) will cause errors or silently corrupt model training. In this dataset the result is zero nulls across all columns — the dataset is clean and no imputation is required.

---

### 4. 📦 Outlier Analysis — Box Plots

```python
plt.figure(figsize=(12, 5))
plt.subplot(1,3,1); sns.boxplot(advertising['TV'])
plt.subplot(1,3,2); sns.boxplot(advertising['Newspaper'])
plt.subplot(1,3,3); sns.boxplot(advertising['Radio'])
```

**What it does:**  
Renders three side-by-side box plots — one per advertising channel — in a single figure.

**How a box plot works:**
- The **box** spans the IQR (Q1 to Q3) — the middle 50% of the data.
- The **line inside the box** is the median (Q2).
- The **whiskers** extend to 1.5× IQR beyond Q1 and Q3.
- Points **beyond the whiskers** are plotted as individual dots — these are statistical outliers.

**Why it matters:** Outliers can disproportionately pull a linear regression line, inflating or deflating the slope. Identifying them early lets you decide whether to remove, cap, or keep them.

---

### 5. 🔍 Exploratory Data Analysis (EDA)

#### 5a. Sales Box Plot

```python
sns.boxplot(advertising['Sales'])
plt.show()
```

Specifically examines the distribution and outliers in the **target variable** (`Sales`). A healthy, roughly symmetric box suggests no major skew in what the model is trying to predict.

---

#### 5b. Pair Plot (Scatter Matrix)

```python
sns.pairplot(advertising,
             x_vars=['TV', 'Newspaper', 'Radio'],
             y_vars='Sales',
             height=4, aspect=1,
             kind='scatter')
plt.show()
```

**What it does:**  
Creates scatter plots of each predictor (TV, Newspaper, Radio) against the target (Sales) in a single figure.

**What to look for:**
- **Linear trend** → the predictor is a good candidate for linear regression.
- **Curved trend** → polynomial or non-linear model may be needed.
- **No trend / random scatter** → the predictor has little predictive power.

**Key insight from this plot:** `TV` shows the strongest, most consistent linear relationship with `Sales` — this is why it was chosen as the sole predictor in the Simple Linear Regression model.

---

#### 5c. Correlation Heatmap

```python
sns.heatmap(advertising.corr(), cmap="YlGnBu", annot=True)
plt.show()
```

**What it does:**  
Computes the **Pearson correlation coefficient** between every pair of columns and displays them as a color-coded matrix.

- Values close to **+1** → strong positive linear relationship.
- Values close to **-1** → strong negative linear relationship.
- Values close to **0** → no linear relationship.

`annot=True` prints the exact coefficient inside each cell. `cmap="YlGnBu"` applies a yellow-green-blue gradient — darker = stronger correlation.

**Key insight:** TV–Sales correlation is the highest (≈ **0.78**), confirming TV spend is the best single predictor of Sales among the three channels.

---

### 6. ✂️ Feature & Target Selection

```python
x = advertising['TV']
y = advertising['Sales']
```

Selects the independent variable (`TV` spend) and the dependent/target variable (`Sales`) for the regression model.

---

### 7. 🔀 Train-Test Split

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

**What it does:**  
Randomly divides the 200 data points into:
- **Training set (80% = 160 samples)** — used to fit the model.
- **Test set (20% = 40 samples)** — held out to evaluate generalization.

**Parameters explained:**
- `test_size=0.2` → 20% of data goes to testing.
- `random_state=42` → Seeds the random number generator so the split is **reproducible** across runs. The value `42` is a widely used convention; any integer works.

**Why it matters:** Evaluating a model on the same data it was trained on gives an overly optimistic picture. The held-out test set simulates "new, unseen data" — a more honest measure of real-world performance.

---

### 8. 🏗️ OLS Model Building with Statsmodels

#### 8a. Adding a Constant (Intercept)

```python
import statsmodels.api as sm
x_train_sm = sm.add_constant(x_train)
```

**What it does:**  
Prepends a column of ones to `x_train`.

**Why it matters:**  
Statsmodels' `OLS` does **not** automatically include an intercept (unlike scikit-learn). Without this step, the regression line is forced through the origin (`y = m·x`), which is almost never correct. Adding the constant allows the model to learn both:
- **Intercept (β₀)** — the predicted Sales when TV = 0.
- **Slope (β₁)** — the increase in Sales for each additional $1K spent on TV.

---

#### 8b. Fitting the OLS Model

```python
lr = sm.OLS(y_train, x_train_sm).fit()
```

**What it does:**  
`OLS` = **Ordinary Least Squares** — the most fundamental linear regression algorithm.

**How OLS works mathematically:**  
It finds the line `ŷ = β₀ + β₁·x` that **minimizes the Sum of Squared Residuals (SSR)**:

```
SSR = Σ (yᵢ - ŷᵢ)²
```

The closed-form solution is: **β = (XᵀX)⁻¹ Xᵀy**

This solution is optimal — it finds the unique line that has the smallest total squared error across all training points (guaranteed by the Gauss-Markov theorem under standard assumptions).

---

#### 8c. Model Parameters

```python
lr.params
```

Returns the fitted coefficients:
- `const` → **β₀ ≈ 7.1196** (intercept: baseline Sales when TV budget = 0)
- `TV` → **β₁ ≈ 0.0465** (slope: for every $1K more spent on TV, Sales increase by ~46.5 units)

---

#### 8d. Statistical Summary

```python
print(lr.summary())
```

Prints a rich statistical report including:

| Metric | Meaning |
|--------|---------|
| **R²** | Proportion of variance in Sales explained by TV (closer to 1 = better fit) |
| **Adj. R²** | R² penalized for extra predictors (same as R² here since we have one predictor) |
| **F-statistic** | Tests whether the overall model is statistically significant |
| **p-value (TV)** | Probability of observing this slope by chance if H₀ is true. p < 0.05 → TV spend is a statistically significant predictor |
| **coef** | Estimated slope and intercept values |
| **std err** | Standard error of each coefficient |
| **t-statistic** | Coefficient divided by its standard error |
| **[0.025, 0.975]** | 95% Confidence Interval for each coefficient |
| **AIC / BIC** | Information criteria for model comparison (lower = better) |

---

### 9. 📉 Visualizing the Regression Line

```python
plt.scatter(x_train, y_train)
plt.plot(x_train, 7.1196 + 0.0465 * x_train, 'r')
plt.show()
```

**What it does:**  
- The **blue dots** represent actual training data points.
- The **red line** is the fitted OLS regression line: `Sales = 7.1196 + 0.0465 × TV`.

Visually inspecting the fit confirms whether the line captures the trend or systematically misses clusters of points.

---

### 10. 📐 Residual Analysis — Error Term Distribution

```python
y_train_pred = lr.predict(x_train_sm)
res = y_train - y_train_pred

sns.distplot(res, bins=15)
```

**What it does:**  
Computes **residuals** (the difference between actual and predicted values) and plots their distribution.

**Why it matters — OLS Assumptions:**  
For OLS regression to be valid, residuals must be:
1. **Normally distributed** → the distribution plot should resemble a bell curve.
2. **Centered around zero** → no systematic over- or under-prediction.
3. **Homoscedastic** → constant variance (not fan-shaped).

A roughly bell-shaped, zero-centered residual plot confirms these assumptions hold and the model is well-specified.

---

### 11. 🔮 Predictions on the Test Set

```python
x_test_sm = sm.add_constant(x_test)
y_pred = lr.predict(x_test_sm)
```

Applies the same `add_constant` step to the test features (critical — must match the training input format), then generates predictions on the 40 held-out samples.

---

### 12. 📏 Model Evaluation — MSE & RMSE

```python
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
```

**Mean Squared Error (MSE):**

```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```

Averages the squared differences between actual and predicted Sales. Squaring penalizes large errors more heavily.

**Root Mean Squared Error (RMSE):**

```
RMSE = √MSE
```

Takes the square root of MSE to restore units back to the original scale (thousands of units). RMSE is the most interpretable regression error metric — it directly answers: *"On average, how far off are the predictions?"*

A lower RMSE indicates better predictive accuracy. RMSE is preferred over MSE for reporting because it's in the same unit as the target variable.

---

### 13. 📈 Test Set Regression Line Visualization

```python
plt.scatter(x_test, y_test)
plt.plot(x_test, 7.1196 + 0.0465 * x_test, 'r')
plt.show()
```

Final visual confirmation: the scatter of test data points against the fitted regression line. If the line passes cleanly through the cloud of test points, it confirms the model generalizes well beyond the training set.

---

## 📐 Final Model Equation

```
Sales = 7.1196 + 0.0465 × TV
```

| Term   | Value  | Interpretation                                                    |
|--------|--------|-------------------------------------------------------------------|
| β₀     | 7.1196 | Baseline Sales (~7,120 units) even with zero TV advertising spend |
| β₁     | 0.0465 | Each additional $1,000 on TV → ~46.5 more units sold             |

---

## ▶️ How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/SALES_PREDICTION_SLR.git
cd SALES_PREDICTION_SLR
```

**2. Install dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

**3. Launch the notebook:**
```bash
jupyter notebook Sales_prediction.ipynb
```

**4. Run all cells sequentially** (Kernel → Restart & Run All).

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| Algorithm | Simple Linear Regression (OLS) |
| Predictor | TV advertising spend |
| Train / Test Split | 80% / 20% |
| R² (Training) | ~0.61 |
| RMSE (Test Set) | ~3.2 (thousands of units) |

---

## 🔮 Future Improvements

- Extend to **Multiple Linear Regression** — include Radio and Newspaper as additional predictors.
- Apply **Polynomial Regression** if non-linear trends exist.
- Perform **Feature Scaling** (StandardScaler) for comparison.
- Add **Cross-Validation** (K-Fold) for more robust evaluation.
- Check all **OLS assumptions** (Durbin-Watson for autocorrelation, Q-Q plot for normality, residual vs. fitted for homoscedasticity).
- Experiment with **Ridge / Lasso Regression** for regularization.

---

## 👤 Author

**Vydhyam Vishnu Sai**  
📧 vishnusaivydhyam@gmail.com
🔗 https://www.linkedin.com/in/vishnusai-vydhyam/ 

