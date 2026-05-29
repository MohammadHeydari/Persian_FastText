# Persian Word Embeddings with FastText

This repository demonstrates how to use **pre-trained FastText embeddings** for Persian (Farsi) and visualize them using **t-SNE** and **Plotly**.

The project focuses on exploring semantic relationships between common Persian words by projecting high-dimensional word vectors into a 2D space.

![tsne Visualization of Persian Words Embedding](https://github.com/MohammadHeydari/Persian_FastText/blob/master/tsne%20Visualization%20of%20Persian%20Words%20Embedding.png)

---

## Features

* Uses **pre-trained FastText Persian embeddings** (`cc.fa.300.bin`)
* Extracts word vectors for a list of common Persian words
* Applies **t-SNE** for dimensionality reduction
* Interactive visualization using **Plotly**
* Simple and educational pipeline for NLP beginners

---

## Requirements

Make sure you have the following installed:

* Python 3.x
* fasttext
* scikit-learn
* plotly
* numpy

---

## Installation

### 1. Clone FastText repository and install

```bash
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
cd ..
```

### 2. Install Python dependencies

```bash
pip install scikit-learn plotly numpy
```

---

## Download Pre-trained Persian Model

```python
import fasttext.util
fasttext.util.download_model('fa', if_exists='ignore')
```

This downloads:

```
cc.fa.300.bin
```

---

## How It Works

### 1. Load FastText model

```python
import fasttext
ft = fasttext.load_model('cc.fa.300.bin')
```

---

### 2. Load Persian words

```python
with open('827_common_parsi_words.txt', 'r') as f:
    words = [w.strip() for w in f.readlines()]
```

---

### 3. Get word embeddings

```python
vecs = [ft.get_word_vector(w) for w in words]
```

Each word is represented as a **300-dimensional vector**.

---

### 4. Apply t-SNE

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=20000, verbose=1)
results = tsne.fit_transform(vecs)
```

---

### 5. Visualize with Plotly

```python
import plotly.offline as py
import plotly.graph_objs as go

plots = []
for i in range(len(words)):
    plots.append(
        go.Scatter(
            x=[results[i, 0]],
            y=[results[i, 1]],
            mode='markers+text',
            text=[words[i]],
            textposition='bottom center'
        )
    )

py.plot(plots, filename='tsne_persianwords.html', auto_open=True)
```

---

## Output

* Interactive HTML visualization:
  `tsne_persianwords.html`

* Static image example:
  `tsne Visualization of Persian Words Embedding.png`

---

## Use Cases

* Exploring semantic similarity in Persian
* NLP education and demonstrations
* Visualizing word embeddings
* Linguistic analysis of Persian vocabulary

---

## Notes

* t-SNE is sensitive to parameters like `perplexity` and `n_iter`
* Results may vary slightly between runs
* Random shuffling of words does **not** affect embedding quality

---

## Contributing

Pull requests are welcome!

For major changes, please open an issue first to discuss what you would like to improve or add.

---

## Future Improvements

* Add clustering (KMeans / HDBSCAN)
* Support for larger vocabularies
* Better UI (Dash / Streamlit)
* Comparison with other embeddings (Word2Vec, BERT)

---
