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

# MyST Markdown

MyST Markdown is a nice extension to the traditional markdown. You can utilize it by adding a header to the `.md` files as shown here. You then have many new features:

- Refer to other documents (chapters) within the book: {doc}`notebook`.
- Cite references stored in a `references.bib` file: {cite}`Rokem2023`.
- Run Python code in this file:

```{code-cell}
print(2 + 2)
```

- Show a figure but hide the code:

```{code-cell} ipython3
---
tags:
    - "remove-input"
mystnb:
  image:
    width: 400px
    alt: Joint distribution of house square footage (x-axis) and house price (y-axis) with marginal distributions on the side
    align: center
  figure:
    name: joint-distq
---

import pandas as pd
import numpy as np
import seaborn as sns

NUM_POINTS = 1000
SEED = 43
X_SCALE = 2000
Y_NOISE_SCALE = 60000

def f(x):
    return 100 + (0.5 * x + 1.5 * x * x) / 100

np.random.seed(SEED)
x = np.random.randn(NUM_POINTS) + 2
x = np.clip(x, 1, 20)
x = X_SCALE * x
y = f(x) + np.random.randn(NUM_POINTS) * Y_NOISE_SCALE
y = y + abs(y.min()) + 10000
df = pd.DataFrame({"Sq. Ft.": x, "Price": y})

_ = sns.jointplot(data = df, x = "Sq. Ft.", y = "Price", kind="kde")
```

- Hide the code but show the output:

```{code-cell} ipython3
:tags: [remove-input]
print("Hello world")
```

- Create a dropdown that hides the code, but it can be elarged again:

```{code-cell} ipython3
---
tags:
  - "hide-input"
mystnb:
  image:
    width: 100%
    align: center
---

print("Hello world")
```

- Add a centered equation with standard LaTeX notation:

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_k x_{in} + \epsilon_i$$


- Add nice colorful boxes. Possible classes are:
  - blue: note, important
  - green: hint, seealso, tip
  - yellow: attention, caution, warning
  - red: danger, error

```{admonition} Summary
:class: tip

Hello world
```

- Generate a table of contents:

```{tableofcontents}
```

- Add a nice symbol to the heading:

## <i class="fas fa-book fa-fw"></i> Assessing Performance

- Create nice references next to the text:

```{margin}
{{ref_test}}\. This is a ref next to the text. Rt can also include things like equations:
$y_i = w_0 + w_1x_i + \varepsilon_i$
```

We can then cite this reference note in text <sup>{{ref_test}}</sup>. Note that this requires you to specify the numbering in the header of the markdown file.

- Add videos:

```{video} https://www.youtube.com/watch?v=zUxOdq3sAFU
```

- Create an empty space:

$\ $

- Display a quiz:

```{code-cell}
from jupyterquiz import display_quiz
display_quiz('quiz/quiz_solution.json')
```

- Or a flash card:

```{code-cell}
from jupytercards import display_flashcards
display_flashcards('quiz/flashcard.json')
```

- We can even open a Python IDE:

```{code-cell}
from IPython.display import IFrame
IFrame('https://trinket.io/embed/python3/3fe4c8f3f4', 700, 500)
```

- Or within text with a different layout and pre-defined code:

<iframe src="https://trinket.io/embed/python3/09d06157a6" width="100%" height="600" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

- You can embed YouTube videos:

````{tab-set}
```{tab-item} Andy's Brain Book
<iframe width="560" height="315" src="https://www.youtube.com/embed/zUxOdq3sAFU?si=CFVUnwIgQiB4jlbz" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
```

```{tab-item} SPM Tutorial
<iframe width="560" height="315" src="https://www.youtube.com/embed/qbcBLXJhzZg?si=qbLQDBgk6lbEXw-O" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
```
````

- You can embed slides:

<iframe width="560" height="315" src="https://mfr.ca-1.osf.io/render?url=https://osf.io/sqcvz/?direct%26mode=render%26action=download%26mode=render" frameborder="0" allowfullscreen></iframe>

- Interactive plots (not working, I now started to use plotly):

```{code-cell}
import ipywidgets as widgets
from ipywidgets import interact

# Create some sample data
np.random.seed(42)
x = np.linspace(-5, 5, 30)
noise = np.random.normal(0, 2, 100)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 50
df = pd.DataFrame({'x': x, 'y': y})

# Define a function that updates the plot based on the selected polynomial order
def update_plot(Order=10):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(data=df, x="x", y="y", ax=ax, order=Order, ci=None)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)
    plt.show()

# Create an interactive slider for the polynomial order
interact(update_plot, Order=widgets.IntSlider(min=1, max=50, step=1, value=1));

```

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Parameters
n_steps = 30               
initial_seed = 42 
orders = [1, 3, 10]
titles = ["1st order model", "3rd order model", "10th order model"]

# Fixed x values and x_fit for regression curve evaluation
x = np.linspace(-3, 3, 30)
x_fit = np.linspace(-3, 3, 400)

# Precompute data for each slider step (each seed)
seed_data = {}
for i in range(n_steps):
    seed = initial_seed + i
    np.random.seed(seed)
    # Generate new y data based on the current seed
    y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 10
    seed_data[seed] = {}
    for order in orders:
        coeffs = np.polyfit(x, y, order)
        y_fit = np.polyval(coeffs, x_fit)
        seed_data[seed][order] = {'y': y, 'y_fit': y_fit}

# Create the figure with 3 subplots (one per model order)
fig = make_subplots(rows=1, cols=3, subplot_titles=titles)

# For the initial slider step (seed = initial_seed), add scatter and regression traces for each subplot.
current_seed = initial_seed
for j, order in enumerate(orders):
    # Scatter trace: raw data
    scatter_trace = go.Scatter(
        x=x,
        y=seed_data[current_seed][order]['y'],
        mode='markers',
        marker=dict(size=10, color='lightgrey', line=dict(color='gray', width=2)),
        showlegend=False
    )
    # Regression line trace
    line_trace = go.Scatter(
        x=x_fit,
        y=seed_data[current_seed][order]['y_fit'],
        mode='lines',
        line=dict(width=3, color='#4c72b0'),
        showlegend=False
    )
    fig.add_trace(scatter_trace, row=1, col=j+1)
    fig.add_trace(line_trace, row=1, col=j+1)

# Create slider steps.
steps = []
for i in range(n_steps):
    seed = initial_seed + i
    new_y = [
        seed_data[seed][orders[0]]['y'],
        seed_data[seed][orders[0]]['y_fit'],
        seed_data[seed][orders[1]]['y'],
        seed_data[seed][orders[1]]['y_fit'],
        seed_data[seed][orders[2]]['y'],
        seed_data[seed][orders[2]]['y_fit']
    ]
    step = dict(
        method="restyle",
        args=[{"y": new_y}],
        label=f"{seed}"
    )
    steps.append(step)

# Define the slider
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Random Seed: "},
    pad={"t": 50},
    steps=steps
)]

# Update the layout: setting axis ranges and adding the slider
fig.update_layout(
    sliders=sliders,
    height=400,
    autosize=True,
    margin=dict(l=10, r=10, t=50, b=20),
)

# Adjust x and y axes ranges for each subplot
for col in range(1, 4):
    fig.update_xaxes(title_text="x", range=[-3.5, 3.5], row=1, col=col, fixedrange=True, gridwidth=1, zerolinewidth=1)
    fig.update_yaxes(title_text="y", range=[-4, 4], row=1, col=col, fixedrange=True, gridwidth=1, zerolinewidth=1)

fig.show()
```