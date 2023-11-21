import textwrap 
from pathlib import Path
from typing import Optional

import srsly
import spacy 
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from prodigy import recipe
from prodigy.util import log
from prodigy.components.stream import Stream
from prodigy.recipes.textcat import manual as textcat_manual
from prodigy.recipes.ner import manual as ner_manual
from prodigy.recipes.spans import manual as spans_manual

from prodigy_ann.util import batched, setup_index, new_text_example_stream

HTML = """
<link
  rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/all.min.css"
  integrity="sha512-1sCRPdkRXhBV2PBLUdRb4tMg1w2YPf37qatUFeS7zlBy7jJI8Lf4VHwWfZZfpXtYSLy85pkm9GaYVYMfw5BC1A=="
  crossorigin="anonymous"
  referrerpolicy="no-referrer"
/>
<details>
    <summary id="reset">Reset stream?</summary>
    <div class="prodigy-content">
        <label class="label" for="query">New query for ANN:</label>
        <input class="prodigy-text-input text-input" type="text" id="query" name="query" value="">
        <br><br>
        <button id="refreshButton" onclick="refreshData()">
            Refresh Stream
            <i
                id="loadingIcon"
                class="fa-solid fa-spinner fa-spin"
                style="display: none;"
            ></i>
        </button>
    </div>
</details>
"""

# We need to dedent in order to prevent a bunch of whitespaces to appear.
HTML = textwrap.dedent(HTML).replace("\n", "")

CSS = """
.inner-div{
  border: 1px solid #ddd;
  text-align: left;
  border-radius: 4px;
}

.label{
  top: -3px;
  opacity: 0.75;
  position: relative;
  font-size: 12px;
  font-weight: bold;
  padding-left: 10px;
}

.text-input{
  width: 100%;
  border: 1px solid #cacaca;
  border-radius: 5px;
  padding: 10px;
  font-size: 20px;
  background: transparent;
  font-family: "Lato", "Trebuchet MS", Roboto, Helvetica, Arial, sans-serif;
}

#reset{
  font-size: 16px;
}
"""

JS = """
function refreshData() {
  document.querySelector('#loadingIcon').style.display = 'inline-block'
  event_data = {
    query: document.getElementById("query").value 
  }
  window.prodigy
    .event('stream-reset', event_data)
    .then(updated_example => {
      console.log('Updating Current Example with new data:', updated_example)
      window.prodigy.update(updated_example)
      document.querySelector('#loadingIcon').style.display = 'none'
    })
    .catch(err => {
      console.error('Error in Event Handler:', err)
    })
}
"""

@recipe(
    "ann.text.index",
    # fmt: off
    source=("Path to text source to index", "positional", None, str),
    index_path=("Path of trained index", "positional", None, Path),
    # fmt: on
)
def text_index(source: Path, index_path: Path):
    """Builds an HSNWLIB index on example text data."""
    # Store sentences as a list, not perfect, but works.
    log("RECIPE: Calling `ann.text.index`")
    examples = [ex["text"] for ex in srsly.read_jsonl(source)]

    # Setup index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = setup_index(model, size=len(examples))

    # Index everything, progbar and save
    iter_examples = tqdm(examples, desc="indexing")
    for batch in batched(iter_examples, n=256):
        embeddings = model.encode(batch)
        index.add_items(embeddings)
    
    # Hnswlib demands a string as an output path
    index.save_index(str(index_path))
    log(f"RECIPE: Index stored in {index_path}")


def stream_reset_calback(examples, index_path):
    def stream_reset(ctrl, *, query: str):
        log(f"RECIPE: Stream reset with query: {query}")
        new_examples = new_text_example_stream(examples, index_path, query=query, n=200)
        ctrl.stream = Stream.from_iterable(new_examples)
        return next(ctrl.stream)
    return stream_reset


@recipe(
    "ann.text.fetch",
    # fmt: off
    source=("Path to text source that has been indexed", "positional", None, str),
    index_path=("Path to index", "positional", None, Path),
    out_path=("Path to write examples into", "positional", None, Path),
    query=("ANN query to run", "option", "q", str),
    n=("Number of results to return", "option", "n", int),
    # fmt: on
)
def text_fetch(source: Path, index_path: Path, out_path: Path, query:str, n:int=200):
    """Fetch a relevant subset using a HNSWlib index."""
    log("RECIPE: Calling `ann.text.fetch`")
    if not query:
        raise ValueError("must pass query")

    stream = new_text_example_stream(source, index_path, query=query, n=n)
    srsly.write_jsonl(out_path, stream)
    log(f"RECIPE: New stream stored at {out_path}")


@recipe(
    "textcat.ann.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    exclusive=("Labels are exclusive", "flag", "e", bool),
    allow_reset=("Allow the user to restart the query", "flag", "r", bool)
    # fmt: on
)
def textcat_ann_manual(
    dataset: str,
    examples: Path,
    index_path: Path,
    labels:str,
    query:str,
    exclusive:bool = False,
    allow_reset: bool = False
):
    """Run textcat.manual using a query to populate the stream."""
    log("RECIPE: Calling `textcat.ann.manual`")
    stream = new_text_example_stream(examples, index_path, query)
    components = textcat_manual(dataset, stream, label=labels.split(","), exclusive=exclusive)
    
    # Only update the components if the user wants to allow the user to reset the stream
    if allow_reset:
        blocks = [
            {"view_id": components["view_id"]}, 
            {"view_id": "html", "html_template": HTML}
        ]
        components["event_hooks"] = {
            "stream-reset": stream_reset_calback(examples, index_path)
        }
        components["view_id"] = "blocks"
        components["config"]["javascript"] = JS
        components["config"]["global_css"] = CSS
        components["config"]["blocks"] = blocks
    return components


@recipe(
    "ner.ann.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    nlp=("spaCy model to load", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    allow_reset=("Allow the user to restart the query", "flag", "r", bool)
    # fmt: on
)
def ner_ann_manual(
    dataset: str,
    nlp: str,
    examples: Path,
    index_path: Path,
    labels:str,
    query:str,
    allow_reset:bool = False,
):
    """Run ner.manual using a query to populate the stream."""
    log("RECIPE: Calling `ner.ann.manual`")
    if "blank" in nlp:
        spacy_mod = spacy.blank(nlp.replace("blank:", ""))
    else:
        spacy_mod = spacy.load(nlp)
    stream = new_text_example_stream(examples, index_path, query)
    # Only update the components if the user wants to allow the user to reset the stream
    components = ner_manual(dataset, spacy_mod, stream, label=labels.split(","))
    if allow_reset:
        blocks = [
            {"view_id": components["view_id"]}, 
            {"view_id": "html", "html_template": HTML}
        ]
        components["event_hooks"] = {
            "stream-reset": stream_reset_calback(examples, index_path)
        }
        components["view_id"] = "blocks"
        components["config"]["javascript"] = JS
        components["config"]["global_css"] = CSS
        components["config"]["blocks"] = blocks
    return components


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
    allow_reset=("Allow the user to restart the query", "flag", "r", bool)
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
    allow_reset: bool = False
):
    """Run spans.manual using a query to populate the stream."""
    log("RECIPE: Calling `spans.ann.manual`")
    if "blank" in nlp:
        spacy_mod = spacy.blank(nlp.replace("blank:", ""))
    else:
        spacy_mod = spacy.load(nlp)
    stream = new_text_example_stream(examples, index_path, query)
    # Only update the components if the user wants to allow the user to reset the stream
    components = spans_manual(dataset, spacy_mod, stream, label=labels.split(","), patterns=patterns)
    if allow_reset:
        blocks = [
            {"view_id": components["view_id"]}, 
            {"view_id": "html", "html_template": HTML}
        ]
        components["event_hooks"] = {
            "stream-reset": stream_reset_calback(examples, index_path)
        }
        components["view_id"] = "blocks"
        components["config"]["javascript"] = JS
        components["config"]["global_css"] = CSS
        components["config"]["blocks"] = blocks
    return components
