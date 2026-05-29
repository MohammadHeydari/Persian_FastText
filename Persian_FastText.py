import os
import random
import fasttext
import fasttext.util
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.manifold import TSNE


# CONFIG
WORDS_FILE = "827_common_parsi_words.txt"
MODEL_LANG = "fa"
MODEL_FILE = "cc.fa.300.bin"
OUTPUT_HTML = "tsne_persianwords.html"


# STEP 1: Download model
def download_model():
    print("Downloading FastText Persian model...")
    fasttext.util.download_model(MODEL_LANG, if_exists="ignore")
    print("Model ready.")


# STEP 2: Load model
def load_model():
    print("Loading FastText model...")
    model = fasttext.load_model(MODEL_FILE)
    print(f"Model loaded. Dimension: {model.get_dimension()}")
    return model


# STEP 3: Load words
def load_words(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")

    with open(file_path, "r", encoding="utf-8") as f:
        words = [w.strip() for w in f.readlines()]

    random.shuffle(words)
    print(f"{len(words)} words loaded.")
    return words


# STEP 4: Generate embeddings
def get_vectors(model, words):
    print("Generating word vectors...")
    vectors = [model.get_word_vector(w) for w in words]
    return vectors


# STEP 5: t-SNE
def apply_tsne(vectors):
    print("Applying t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=20000,
        verbose=1
    )
    results = tsne.fit_transform(vectors)
    return results


# STEP 6: Visualization
def visualize(words, coords):
    print("Creating visualization...")

    plots = []
    for i in range(len(words)):
        plots.append(
            go.Scatter(
                x=[coords[i, 0]],
                y=[coords[i, 1]],
                mode="markers+text",
                text=[words[i]],
                textposition="bottom center",
                marker=dict(
                    size=10,
                    color=i,
                    colorscale="Jet",
                    opacity=0.8
                ),
                textfont=dict(size=14),
                name=words[i]
            )
        )

    py.plot(plots, filename=OUTPUT_HTML, auto_open=True)
    print(f"Saved to {OUTPUT_HTML}")


# MAIN PIPELINE
def main():
    download_model()
    model = load_model()
    words = load_words(WORDS_FILE)
    vectors = get_vectors(model, words)
    coords = apply_tsne(vectors)
    visualize(words, coords)


if __name__ == "__main__":
    main()
```
