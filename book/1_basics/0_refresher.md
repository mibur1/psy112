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

# <i class="fa-solid fa-repeat"></i> Recap: Regression

One of the the most important concepts you learned about in the [psy111 seminar](https://mibur1.github.io/psy111) were (linear) regression models. Let's quickly recap this concept and how to implement it in Python.

Have a look at the following data:



```{code-cell} ipython3
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Generate sample data
np.random.seed(42)
x = np.linspace(-5, 5, 30)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 50
df = pd.DataFrame({'x': x, 'y': y})

# Create scatter trace for the raw data
scatter = go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    marker=dict(
        size=10,
        color='lightgrey',
        line=dict(color='gray', width=2)
    ),
    name='Data'
)

# Fit a linear regression model (polynomial order 1)
coeffs = np.polyfit(df['x'], df['y'], 1)
x_fit = np.linspace(-5, 5, 400)
y_fit = np.polyval(coeffs, x_fit)

# Create regression trace with a thicker blue line
regression = go.Scatter(
    x=x_fit,
    y=y_fit,
    mode='lines',
    name='Model',
    line=dict(width=3, color='#4c72b0')
)

# Combine the traces into one figure
data = [scatter, regression]

# Define layout without annotations or sliders
layout = go.Layout(
    xaxis=dict(range=[-5, 5], tickfont=dict(size=14)),
    yaxis=dict(range=[-3, 3], tickfont=dict(size=14)),
    legend=dict(font=dict(size=14)),
    margin=dict(l=10, r=10, t=30, b=20)
)

# Create the figure
fig = go.Figure(data=data, layout=layout)
fig

```

Let's have a look at the model statistics:

```{code-cell} ipython3
import statsmodels.formula.api as smf

model = smf.ols("y ~ x", data=df).fit()
print(model.summary())
```

You can see that we have a R² of 0.753, which means that our linear regression model explains 75% of the variance in our data. That's pretty good! But I'm sure we can do better. After all, life is more complicated than just a straigth line, no?


```{code-cell} ipython3
:tags: [remove-input]

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Suppress polyfit conditioning warnings
warnings.filterwarnings("ignore", message=".*Polyfit may be poorly conditioned.*")

# Generate sample data
np.random.seed(42)
x = np.linspace(-5, 5, 30)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 50
df = pd.DataFrame({'x': x, 'y': y})

# Scatter trace for the raw data with updated marker properties
scatter = go.Scatter(x=df['x'], y=df['y'], mode='markers',
                     marker=dict(size=10, color='lightgrey', line=dict(color='gray', width=2)), name='Data') 

# Generate regression curves for polynomial orders 1 through 30 with updated line properties
regression_traces = []
r2_list = []
x_fit = np.linspace(-5, 5, 400)
for order in range(1, 31):
    coeffs = np.polyfit(df['x'], df['y'], order)
    y_fit = np.polyval(coeffs, x_fit)
    trace = go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Model', visible=False,
                       line=dict(width=3, color='#4c72b0'))
    regression_traces.append(trace)

    y_pred = np.polyval(coeffs, df['x'])
    r2 = 1 - np.sum((df['y'] - y_pred) ** 2) / np.sum((df['y'] - df['y'].mean()) ** 2)
    r2_list.append(r2)

# Combine data: always show the scatter trace, and one regression trace at a time (set first one visible)
data = [scatter] + regression_traces
data[1]['visible'] = True

# Create slider steps
steps = []
for i in range(30):
    vis = [True] + [False] * 30
    vis[i + 1] = True
    step = dict(
        method="update",
        args=[{"visible": vis},
              {"annotations": [dict(x=0.02, y=0.98, xref="paper", yref="paper", text=f"Model R² = {r2_list[i]:.3f}", 
                                    showarrow=False, font=dict(size=18, color="gray"), align="left")]}],
        label=str(i + 1)
    )
    steps.append(step)

# Define the slider
sliders = [dict(active=0, currentvalue={"prefix": "Order of the polynomial regression model: "}, pad={"t": 30}, steps=steps)]

# Define the layout with
layout = go.Layout(
    annotations=[dict(x=0.02, y=0.98, xref="paper", yref="paper", text=f"Model R² = {r2_list[0]:.3f}",
                      showarrow=False, font=dict(size=18, color="gray"), align="left")],
    sliders=sliders,
    xaxis=dict(range=[-5, 5], tickfont=dict(size=14)),
    yaxis=dict(range=[-3, 3], tickfont=dict(size=14)),
    legend=dict(font=dict(size=14)),
    margin=dict(l=10, r=10, t=30, b=20)
)

# Create the figure
fig = go.Figure(data=data, layout=layout)
fig
```



```{code-cell} ipython3
```



```{code-cell} ipython3

```