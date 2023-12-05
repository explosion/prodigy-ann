from pathlib import Path

import srsly

from prodigy import recipe
from prodigy.util import log
from prodigy.recipes.image import image_manual
from .util import remove_images, ApproximateIndex, JS, CSS, HTML, stream_reset_calback


@recipe(
    "ann.image.index",
    # fmt: off
    source=("Path to text source to index", "positional", None, str),
    index_path=("Path to output the trained index", "positional", None, Path),
    # fmt: on
)
def image_index(source: Path, index_path: Path):
    """Builds an HSNWLIB index on example image data."""
    # Store sentences as a list, not perfect, but works.
    log("RECIPE: Calling `ann.image.index`")
    index = ApproximateIndex('clip-ViT-B-32', source)
    index.build_index(setting="image")
    
    # Hnswlib demands a string as an output path
    index.store_index(index_path)


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

    index = ApproximateIndex('clip-ViT-B-32', source, index_path)
    stream = index.new_stream(query, n=n)
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
    allow_reset=("Allow the user to restart the query", "flag", "r", bool),
    # fmt: on
)
def image_ann_manual(
        dataset: str,
        source: Path,
        index_path: Path,
        labels: str,
        query: str,
        remove_base64: bool = False,
        n: int = 100,
        allow_reset: bool = False,
):
    """Run image.manual using a query to populate the stream."""
    index = ApproximateIndex(model_name='clip-ViT-B-32', source=source, index_path=index_path)
    stream = index.new_stream(query, n=n)
    components = image_manual(dataset, source=stream, loader="images", label=labels.split(","), remove_base64=remove_base64)
    # Only update the components if the user wants to allow the user to reset the stream
    if allow_reset:
        blocks = [
            {"view_id": components["view_id"]}, 
            {"view_id": "html", "html_template": HTML}
        ]
        components["event_hooks"] = {
            "stream-reset": stream_reset_calback(index, n)
        }
        components["view_id"] = "blocks"
        components["config"]["javascript"] = JS
        components["config"]["global_css"] = CSS
        components["config"]["blocks"] = blocks
    return components
