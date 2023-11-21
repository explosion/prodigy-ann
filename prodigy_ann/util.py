import textwrap 
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
from prodigy.util import log
from sentence_transformers import SentenceTransformer


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
    log(f"RECIPE: Loaded index from {path}")
    return index


def new_text_example_stream(source: Path, index_path: Path, query:str, n:int=200) -> List[str]:
    log(f"RECIPE: New query for a new stream: {query}.")
    log(f"RECIPE: Generating new stream from {source} and {index_path}.")
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
