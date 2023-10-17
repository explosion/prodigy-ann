from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Optional

import srsly
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from prodigy import recipe
from prodigy.components.stream import get_stream
from prodigy.recipes.image import image_manual

from prodigy_ann.util import batched, setup_index, load_index, new_example_stream


@recipe(
    "ann.image.index",
    # fmt: off
    source=("Path to text source to index", "positional", None, str),
    index_path=("Path of trained index", "positional", None, Path),
    # fmt: on
)
def image_index(source: Path, index_path: Path):
    """Builds an HSNWLIB index on example text data."""
    # Store sentences as a list, not perfect, but works.
    stream = get_stream(source)
    print(next(stream))

    # Setup index
    model = SentenceTransformer('clip-ViT-B-32')
    index = setup_index(model, size=len(examples))

    # Index everything, progbar and save
    iter_examples = tqdm(examples, desc="indexing")
    for batch in batched(iter_examples, n=256):
        embeddings = model.encode(batch)
        index.add_items(embeddings)

    # Hnswlib demands a string as an output path
    index.save_index(str(index_path))


@recipe(
    "ann.image.fetch",
    # fmt: off
    source=("Path to text source that has been indexed", "positional", None, str),
    index_path=("Path to index", "positional", None, Path),
    out_path=("Path to write examples into", "positional", None, Path),
    query=("ANN query to run", "option", "q", str),
    n=("Number of results to return", "option", "n", int),
    # fmt: on
)
def image_fetch(source: Path, index_path: Path, out_path: Path, query: str, n: int = 200):
    """Fetch a relevant subset using a HNSWlib index."""
    if not query:
        raise ValueError("must pass query")

    # Store sentences as a list, not perfect, but works.
    examples = [ex["text"] for ex in srsly.read_jsonl(source)]

    # Setup index
    model = SentenceTransformer('clip-ViT-B-32')
    index = load_index(model, size=len(examples), path=index_path)
    stream = new_example_stream(examples, index, query=query, model=model, n=n)
    srsly.write_jsonl(out_path, stream)


@recipe(
    "image.ann.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    nlp=("spaCy model to load", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    # fmt: on
)
def ner_ann_manual(
        dataset: str,
        nlp: str,
        examples: Path,
        index_path: Path,
        labels: str,
        query: str,
):
    """Run ner.manual using a query to populate the stream."""
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        image_fetch(examples, index_path, out_path=tmpfile.name, query=query)
        stream = list(srsly.read_jsonl(tmpfile.name))
        image_manual(dataset, nlp, stream, label=labels)


@recipe(
    "image.ann.clf",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    nlp=("spaCy model to load", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    patterns=("Path to match patterns file", "option", "pt", Path),
    query=("ANN query to run", "option", "q", str),
    # fmt: on
)
def image_ann_manual(
        dataset: str,
        nlp: str,
        examples: Path,
        index_path: Path,
        labels: str,
        query: str,
        patterns: Optional[Path] = None,
):
    """Run spans.manual using a query to populate the stream."""
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        image_fetch(examples, index_path, out_path=tmpfile.name, query=query)
        stream = list(srsly.read_jsonl(tmpfile.name))
        spans_manual(dataset, nlp, stream, label=labels, patterns=patterns)
