import itertools as it
from pathlib import Path
from typing import List

from hnswlib import Index
from sentence_transformers import SentenceTransformer


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
