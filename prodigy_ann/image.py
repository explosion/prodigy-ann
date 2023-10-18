from tempfile import NamedTemporaryFile
from pathlib import Path

import srsly
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image

from prodigy import recipe
from prodigy.util import set_hashes
from prodigy.components.stream import get_stream
from prodigy.recipes.image import image_manual

from prodigy_ann.util import batched, setup_index, load_index, new_image_example_stream


def remove_images(examples):
    # Remove all data URIs before storing example in the database
    for eg in examples:
        if eg["image"].startswith("data:"):
            del eg["image"]
        yield set_hashes(eg)

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
    stream.apply(remove_images)
    examples = list(stream)

    # Setup index
    model = SentenceTransformer('clip-ViT-B-32')
    index = setup_index(model, size=len(examples))

    # Index everything, progbar and save
    iter_examples = tqdm(examples, desc="indexing")
    for batch in batched(iter_examples, n=256):
        embeddings = model.encode([Image.open(ex['path']) for ex in batch])
        index.add_items(embeddings)

    # Hnswlib demands a string as an output path
    index.save_index(str(index_path))


@recipe(
    "ann.image.fetch",
    # fmt: off
    source=("Path to image source that has been indexed", "positional", None, str),
    index_path=("Path to index", "positional", None, Path),
    out_path=("Path to write examples into", "positional", None, Path),
    query=("ANN query to run", "option", "q", str),
    n=("Number of results to return", "option", "n", int),
    remove_base64=("Remove base64-encoded image data", "flag", "R", bool)
    # fmt: on
)
def image_fetch(source: Path, index_path: Path, out_path: Path, query: str, n: int = 200, remove_base64:bool=False):
    """Fetch a relevant subset using a HNSWlib index."""
    if not query:
        raise ValueError("must pass query")

    # Store sentences as a list, not perfect, but works.
    stream = get_stream(source)
    stream.apply(remove_images)
    examples = list(stream)

    # Setup index
    model = SentenceTransformer('clip-ViT-B-32')
    index = load_index(model, size=len(examples), path=index_path)
    stream = new_image_example_stream(examples, index, query=query, model=model, n=n)
    if remove_base64:
        stream = remove_images(stream)
    srsly.write_jsonl(out_path, stream)

    # Return stream to make downstreams recipes easy to create
    examples = new_image_example_stream(examples, index, query=query, model=model, n=n)
    return get_stream(examples)



@recipe(
    "image.ann.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    remove_base64=("Remove base64-encoded image data", "flag", "R", bool),
    n=("Number of results to return", "option", "n", int),
    # fmt: on
)
def image_ann_manual(
        dataset: str,
        examples: Path,
        index_path: Path,
        labels: str,
        query: str,
        remove_base64: bool = False,
        n: int = 100
):
    """Run image.manual using a query to populate the stream."""
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        stream = image_fetch(examples, index_path, out_path=tmpfile.name, query=query, n=n)
        return image_manual(dataset, source=stream, loader="images", label=labels.split(","), remove_base64=remove_base64)
