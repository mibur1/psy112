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

