from tempfile import NamedTemporaryFile
import itertools as it
from pathlib import Path
from typing import List, Optional

import srsly
from sentence_transformers import SentenceTransformer
from hnswlib import Index
from tqdm import tqdm

from prodigy import msg, recipe
from prodigy.recipes.textcat import manual as textcat_manual
from prodigy.recipes.ner import manual as ner_manual
from prodigy.recipes.spans import manual as spans_manual



def batched(iterable, n=56):
    "Batch data into tuples of length n. The last batch may be shorter."
    if n < 1:
        raise ValueError("n must be at least one")
    iters = iter(iterable)
    while batch := tuple(it.islice(iters, n)):
        yield batch


def setup_index(model: SentenceTransformer, size:int) -> Index:
    out = model.encode(["Test text right here."])
    index = Index(space="cosine", dim=out.shape[1])
    index.init_index(max_elements=size)
    return index


def load_index(model: SentenceTransformer, size: int, path:Path) -> Index:
    out = model.encode(["Test text right here."])
    index = Index(space="cosine", dim=out.shape[1])
    index.load_index(str(path), max_elements=size)
    return index


def new_example_stream(
        examples: List[str], 
        index:Index, 
        query:str, 
        model:SentenceTransformer, 
        n:int=200
    ):
    """New generator based on query/index/model."""
    embedding = model.encode([query])[0]
    items, distances = index.knn_query([embedding], k=n)

    for lab, dist in zip(items[0].tolist(), distances[0].tolist()):
        ex = {
            "text": str(examples[int(lab)]),
            "meta": {"distance": float(dist), "query": query}
        }
        yield ex


@recipe(
    "ann.text.index",
    # fmt: off
    examples=("Datafile to annotate", "positional", None, str),
    out_path=("Path trained index", "positional", None, Path),
    # fmt: on
)
def index(examples: Path, out_path: Path):
    """Builds an HSNWLIB index on example text data."""
    # Store sentences as a list, not perfect, but works.
    examples = [ex["text"] for ex in srsly.read_jsonl(examples)]

    # Setup index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = setup_index(model, size=len(examples))

    # Index everything, progbar and save
    iter_examples = tqdm(examples, desc="indexing")
    for batch in batched(iter_examples, n=256):
        embeddings = model.encode(batch)
        index.add_items(embeddings)
    
    # Hnswlib demands a string as an output path
    index.save_index(str(out_path))


@recipe(
    "ann.text.fetch",
    # fmt: off
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    out_path=("Path to write examples into", "positional", None, Path),
    query=("ANN query to run", "option", "q", str),
    n=("Number of results to return", "option", "n", int),
    # fmt: on
)
def fetch(examples: Path, index_path: Path, out_path: Path, query:str, n:int=200):
    """Fetch a relevant subset using a pretrained index."""
    if not query:
        raise ValueError("must pass query")
    
    # Store sentences as a list, not perfect, but works.
    examples = [ex["text"] for ex in srsly.read_jsonl(examples)]

    # Setup index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = load_index(model, size=len(examples), path=index_path)
    stream = new_example_stream(examples, index, query=query, model=model, n=n)
    srsly.write_jsonl(out_path, stream)


@recipe(
    "textcat.ann.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    exclusive=("Labels are exclusive", "flag", "e", bool),
    # fmt: on
)
def textcat_ann_manual(
    dataset: str,
    examples: Path,
    index_path: Path,
    labels:str,
    query:str,
    exclusive:bool = False
):
    """Run textcat.manual using a query to populate the stream."""
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        fetch(examples, index_path, out_path=tmpfile.name, query=query)
        stream = list(srsly.read_jsonl(tmpfile.name))
        return textcat_manual(dataset, stream, label=labels.split(","), exclusive=exclusive)


@recipe(
    "ner.ann.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    nlp=("spaCy model to load", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    exclusive=("Labels are exclusive", "flag", "e", bool),
    # fmt: on
)
def ner_ann_manual(
    dataset: str,
    nlp: str,
    examples: Path,
    index_path: Path,
    labels:str,
    query:str,
    exclusive:bool = False
):
    """Run ner.manual using a query to populate the stream."""
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        fetch(examples, index_path, out_path=tmpfile.name, query=query)
        stream = list(srsly.read_jsonl(tmpfile.name))
        ner_manual(dataset, nlp, stream, label=labels, exclusive=exclusive)


@recipe(
    "spans.ann.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    nlp=("spaCy model to load", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    patterns=("Path to match patterns file", "option", "pt", Path),
    query=("ANN query to run", "option", "q", str),
    exclusive=("Labels are exclusive", "flag", "e", bool),
    # fmt: on
)
def spans_ann_manual(
    dataset: str,
    nlp: str,
    examples: Path,
    index_path: Path,
    labels:str,
    query:str,
    patterns: Optional[Path] = None,
    exclusive:bool = False
):
    """Run spans.manual using a query to populate the stream."""
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        fetch(examples, index_path, out_path=tmpfile.name, query=query)
        stream = list(srsly.read_jsonl(tmpfile.name))
        spans_manual(dataset, nlp, stream, label=labels, exclusive=exclusive, patterns=patterns)
