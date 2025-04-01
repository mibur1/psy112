---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
myst:
  substitutions:
    ref_test: 1
---

# <i class="fa-solid fa-puzzle-piece"></i> Regularization

Building on subset selection, an alternative approach is to include all *p* predictors in the model but apply regularization—shrinking the **coefficient estimates toward zero** relative to the least squares estimates. This reduces model complexity without fully discarding variables. Though it introduces some bias, it often lowers variance and improves test performance. 

3 approaches are commonly used to regularize the predictors:
- Ridge Regression (L2 Regression)
- Lasso Regression (L1 Regression)
- Elastic Net Regression


----------------------------------------------------------------
### *Todays data with many predictors - Hitters dataset*
For pracitcal demonstration, we will use again the `Hitters` dataset. 

```{code-cell} 
import statsmodels.api as sm 

# get dataset
hitters = sm.datasets.get_rdataset("Hitters", "ISLR").data

# keeping a total of 15 variables - the target ´Salary´ and 14 features.
hitters_subset = hitters[["Salary", "AtBat", "Runs","RBI", "CHits", "CAtBat", "CRuns", "CWalks", "Assists", "Hits", "HmRun", "Years", "Errors", "Walks"]].copy()

# make sure the rows, containing missing values, are dropped
hitters_subset.dropna(inplace=True)

hitters_subset.head()
```
Let's also take a look at the correlation between predictors to check for potential multicollinearity, which can affect the stability of linear regression models.

```{code-cell} 
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation heatmap for hitters subset
plt.figure(figsize=(8, 5))
sns.heatmap(hitters_subset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Hitters Subset", fontsize=16)
plt.show()
```
The heatmap reveals strong correlations between several predictors, indicating multicollinearity — making Ridge regression an appropriate modeling choice. However, we need to be cautious when correlations are too high, as this suggests that some features are nearly duplicates.
- `CHits` and `CAtBat` show a correlation of *1*,
- `CHits` and `CRuns`have a very strong correlation of *0.98*,
- `Hits` and `AtBat` are also highly correlated, with *0.96* 

In such cases, it may be beneficial to remove one of the correlated features to avoid redundancy and improve model interpretability.

**Hands on:**
Again take a look at what the feautures are measuring (https://islp.readthedocs.io/en/latest/datasets/Hitters.html) and decide, which one to drop! To make your decision not just random, maybe think about the relevance in predicting the `Salary`. Drop the features and check the heatmap again!

<iframe src="https://trinket.io/embed/python3/31e21c627234" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

```{code-cell} ipython3
:tags: [remove-input]
## Removing features
# List of features to drop
features_drop = ["AtBat", "CRuns", "CAtBat"]
# dropping 
hitters_subset2= hitters_subset.drop(columns= features_drop)
```

Then we can continue prepare our data by defining target and features and split it into training and test set. 

```{code-cell} 
from sklearn.model_selection import train_test_split

# Defining target and features
X = hitters_subset2.drop(columns=["Salary"])
y = hitters_subset2["Salary"]

# split into training and test data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Ridge Regression
```{margin}
Lamda is a tuning parameter that controls the strength of the penalty! 
```

$$
\sum_{i=1}^{n}\left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
$$

Where: 
- $ \sum_{i=1}^{n}\left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2  $ is the **residual sum of squares (RSS)**  

- $ \lambda \sum_{j=1}^{p} \beta_j^2 $ is the **L2 penalty** on the coefficients 



```{admonition} The tuning parameter λ
:class: note 

λ controls the degree of regulariztation and the relative impact of the penalty on the parameter estimates.

- λ=0: penalty term has no effect (Ridge Regression will produce least square estimates)
- As λ increases, the impact of the shirnkage penalty grows

 Thus, selecting a good value of lambda is crucial. For this, we can use cross-validation.
```
**Step 1:** 
Before we implement ridge regression, we need to standardize the variables since ridge regression is senstivite to scaling. This can be done by using `StandardScaler`from `scikit-learn`package. 

```{code-cell} 
# scale the data to have mean 0 and stdev 1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform test data
X_test_scaled = scaler.transform(X_test)
# This ensures both sets are standardized consistently, but only the training data 
# is used to compute the scaling parameters.
```

**Step 2:**
Next, we set up a range of values for λ. The graph nicely visualize how the beta values change with increasing lamda. 

```{code-cell} ipython3
:tags: [remove-input]

from sklearn.linear_model import Ridge
import numpy as np 

#initialize list to store coefficient values
coef=[]
alphas = np.linspace(0.001, 20, 80) 

for a in alphas:
  ridgereg=Ridge(alpha=a)
  ridgereg.fit(X_train_scaled,y_train)
  coef.append(ridgereg.coef_)

# Make plot of Beta as a function of Lamda
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(alphas,coef)
ax.set_xlabel('Lambda (Regularization Parameter)')
ax.set_ylabel('Beta (Predictor Coefficients)')
ax.set_title('Ridge Coefficients vs Regularization Parameters')
ax.axis('tight')
```

```{code-cell}
import numpy as np 

# set range 
lambda_range= np.linspace(0.001, 20, 80) 
```

**Step 3:** 
For each value of lambda, ridge regression is performed on the training data. Cross-validation is then used to identify the optimal lambda that minimizes prediction error, and the final model is fitted using this best value.

```{margin}
In 'scikit-learn' the lamda parameter is called alpha! Don't get confused by that.
```

```{code-cell} 
from sklearn.linear_model import RidgeCV
import pandas as pd

# getting the best alpha through cross validation
ridge_cv = RidgeCV(alphas=lambda_range)
# fit ridge regression on given data using best alpha
ridge_cv.fit(X_train_scaled, y_train) 

print(f"The optimal alpha value for our analysis ends up being {ridge_cv.alpha_}.")

# getting R² traing score 
train_score_ridge= ridge_cv.score(X_train_scaled, y_train)

print(f"The R² train score for ridge model is {format(train_score_ridge)}.")
```
For identifying the model coefficients, we can print a table listing those. 
```{code-cell} 
# create a DataFrame to display predictor names and their corresponding coefficients
coef_table = pd.DataFrame({
    'Predictor': X_train.columns,
    'Ridge Coefficient': ridge_cv.coef_
})

# sort table by absolute value of Ridge Coefficient
coef_table = coef_table.reindex(coef_table['Ridge Coefficient'].abs().sort_values(ascending=False).index)

print(coef_table)
```

**Step 4:** 
Finally, we evaluate the model's performance on unseen data to assess its generalization ability.

```{code-cell} 
# fitting it and getting R² test score 
test_score_ridge= ridge_cv.score(X_test_scaled, y_test)

print(f"The R² test score for ridge model is {format(test_score_ridge)}.")
```

### Lasso Regression
Lasso takes the same idea as Ridge Regression and aims to shrinkage coefficients toward zero. In contrast to Ridge Regression Lasso does not include all predcitors in the final model by forcing some predictors to be exactly zero. 

$$
\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j|\
$$

Where: 
- $ \sum_{i=1}^{n}\left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2  $ is the **residual sum of squares (RSS)**  

- $ \lambda \sum_{j=1}^{p} |\beta_j|\ $ is the **L1 penalty** on the coefficients 


<br>
Just as the basic concept, also the implementation of Lasso regression in python is quite similir. As a first step we need to **standardize** the features and as the second step we set up a **range** for our tuning parameter λ. In  third step we identify the **best λ** by applying `LassoCV` from the `scikit-learn` package. Lastly, we can use the test data set to evalute the test performance. 

<br>

**Hands on**: Please use the code chunk below and apply a Lasso Regression on our Hitters subset! Please try to answer the following questions:
- What is the best lamda this time?
- How does the model perform with unseen data?
- Looking at the coefficients, how many predictor were forced to become 0? According to this model, what are the key features? 

<iframe src="https://trinket.io/embed/python3/7f210b6e2f87" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>


```{code-cell} ipython3
#:tags: [remove-input]

# Dataset: hitters_subset2 - already splitted into X and y and splitted into training and test- dataset.
import numpy as np 
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# STEP 1: Standarize 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Use the same scaler to transform test data
X_test_scaled = scaler.transform(X_test)

# STEP 2: Range for lamda
lambda_range= np.linspace(0.001, 20, 80) 

# STEP 3: Identifying best lamda and fit 
# getting the best alpha through cross validation
lasso_cv = LassoCV(alphas=lambda_range)
# fit ridge regression on given data using best alpha
lasso_cv.fit(X_train_scaled, y_train) 

print(f"The optimal alpha value for our analysis ends up being {lasso_cv.alpha_}. The best option for Ridge regression was {ridge_cv.alpha_}")

# getting R² traing score 
train_score_lasso=lasso_cv.score(X_train_scaled, y_train)

print(f"The R² train score for lasso model is {format(train_score_lasso)}.")

# LOOKING AT COEFFICIENTS
# create a DataFrame to display predictor names and their corresponding coefficients
coef_table = pd.DataFrame({
    'Predictor': X_train.columns,
    'Lasso_Coefficient': lasso_cv.coef_
})

# sort table by absolute value of Ridge Coefficient
coef_table = coef_table.reindex(coef_table['Lasso_Coefficient'].abs().sort_values(ascending=False).index)

print(coef_table)

# STEP 4: Evaluationg
# fitting it and getting R² test score 
test_score_lasso= lasso_cv.score(X_test_scaled, y_test)

print(f"The R² test score for lasso model is {format(test_score_lasso)}.")
```

### Lasso or Ridge Regression?! 
In General , neither Righe nor Lasso will universally dominate the other! However, there are 3 main differences: 
1) Regularization Type
    - Lasso regression applies L1 regularization, penalizing the absolute values of the coefficients
    - Ridge regression applies L2 regularization, penalizing the squared values of the coefficients
2) Coefficient Selection 
    -  Lasso tends to shrink some coefficients all the way to zero, effectively performing feature selection by excluding less relevant variables
    - Ridge also shrinks coefficients toward zero, but does not eliminate any completely. Instead of selecting variables, it distributes the effect across all features

```{image} ./figures/Budget_LassoRidge.png 
:alt: Budget Lasso vs Ridge Regression
:width: 80%
:align: center 
```
<br>

3) Response to multiple connilearity:
   - Lasso tends to select one variable from a group of highly correlated predictors and set the others to zero 
    -  Ridge, on the other hand, shares the influence across correlated predictors by shrinking their coefficients without removing any

```{code-cell}  ipython3
:tags: [remove-input]
from sklearn.linear_model import LinearRegression
# having linear regression as comparison
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# PLOTTING
import matplotlib.pyplot as pl
features = hitters_subset2.drop(columns=["Salary"])
# getting feaute names
features = features.columns.tolist()

plt.figure(figsize = (8, 5))

#add plot for ridge regression
plt.plot(features ,ridge_cv.coef_,alpha=0.8,linestyle='none',marker='d',markersize=5,color='green',label=r'Ridge Regression',zorder=7)

#add plot for lasso regression
plt.plot(lasso_cv.coef_,alpha=0.8,linestyle='none',marker='d',markersize=6,color='#8A2BE2',label=r'Lasso Regression')

#add plot for linear model
plt.plot(features,lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='lightpink',label='Linear Regression')

# Horizontal dashed line at y = 0
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.7)

#rotate axis
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.ylim(bottom=-27)
plt.legend()
plt.title("Ridge vs. Lasso Coefficients")
plt.show()
```
