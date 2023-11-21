from pathlib import Path

import srsly
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image

from prodigy import recipe
from prodigy.util import set_hashes, log
from prodigy.components.stream import get_stream
from prodigy.recipes.image import image_manual

from prodigy_ann.util import batched, setup_index, remove_images, new_image_example_stream


@recipe(
    "ann.image.index",
    # fmt: off
    source=("Path to text source to index", "positional", None, str),
    index_path=("Path to output the trained index", "positional", None, Path),
    # fmt: on
)
def image_index(source: Path, index_path: Path):
    """Builds an HSNWLIB index on example text data."""
    # Store sentences as a list, not perfect, but works.
    log("RECIPE: Calling `ann.image.index`")
    stream = get_stream(source)
    stream.apply(remove_images)
    examples = list(stream)

    # Setup index
    model = SentenceTransformer('clip-ViT-B-32')
    index = setup_index(model, size=len(examples))

    # Index everything, progbar and save
    iter_examples = tqdm(examples, desc="indexing")
    for batch in batched(iter_examples, n=64):
        embeddings = model.encode([Image.open(ex['path']) for ex in batch])
        index.add_items(embeddings)

    # Hnswlib demands a string as an output path
    index.save_index(str(index_path))
    log(f"RECIPE: Index stored at {index_path}")


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
    log("RECIPE: Calling `ann.image.fetch`")
    if not query:
        raise ValueError("must pass query")

    # Store sentences as a list, not perfect, but works.
    stream = get_stream(source)
    stream.apply(remove_images)
    examples = list(stream)

    # Setup index
    stream = new_image_example_stream(examples, index_path, query=query, n=n)
    if remove_base64:
        stream = remove_images(stream)
    srsly.write_jsonl(out_path, stream)
    log(f"RECIPE: New stream stored at {out_path}")


@recipe(
    "image.ann.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    source=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    remove_base64=("Remove base64-encoded image data", "flag", "R", bool),
    n=("Number of results to return", "option", "n", int),
    # fmt: on
)
def image_ann_manual(
        dataset: str,
        source: Path,
        index_path: Path,
        labels: str,
        query: str,
        remove_base64: bool = False,
        n: int = 100
):
    """Run image.manual using a query to populate the stream."""
    new_stream = new_image_example_stream(source, index_path, query=query, n=n)
    return image_manual(dataset, source=new_stream, loader="images", label=labels.split(","), remove_base64=remove_base64)
