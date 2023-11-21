import srsly 
from typing import Dict
import base64
from io import BytesIO
from PIL import Image

import itertools as it
from pathlib import Path
from typing import List

from hnswlib import Index
from prodigy.util import set_hashes
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


def new_text_example_stream(source: Path, index_path: Path, query:str, n:int=200) -> List[str]:
    examples = [ex for ex in srsly.read_jsonl(source)]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = load_index(model, size=len(examples), path=index_path)
    embedding = model.encode([query])[0]
    items, distances = index.knn_query([embedding], k=n)

    for lab, dist in zip(items[0].tolist(), distances[0].tolist()):
        # Get the original example
        ex = examples[int(lab)]

        # Add some extra meta info
        ex['meta'] = ex.get("meta", {})
        ex['meta']['index'] = int(lab)
        ex['meta']['distance'] = float(dist)
        ex['meta']["query"] = query

        # Don't forget hashes
        yield set_hashes(ex)


def base64_image(example: Dict) -> str:
    """Turns a PdfPage into a base64 image for Prodigy"""
    pil_image = Image.open(example['path']).convert('RGB')
    with BytesIO() as buffered:
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/png;base64,{img_str.decode('utf-8')}"


def new_image_example_stream(
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
            **examples[int(lab)],
            "meta": {"distance": float(dist), "query": query},
        }
        ex['image'] = base64_image(ex)
        yield ex
