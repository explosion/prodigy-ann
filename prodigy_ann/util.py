import itertools as it
from pathlib import Path
from typing import List, Optional, Callable, Literal
import textwrap
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from hnswlib import Index
from prodigy.util import set_hashes
from prodigy.util import log
from prodigy.components.stream import Stream
from prodigy.components.stream import get_stream
from prodigy.core import Controller

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
      window.prodigy.resetQueue();
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


def add_hashes(examples):
    for ex in examples:
        yield set_hashes(ex)

class ApproximateIndex:
    def __init__(self, model_name:str, source: Path, index_path: Optional[Path] = None):
        log(f"INDEX: Using {model_name=} and source={str(source)}.")
        stream = get_stream(source)

        # Always add the hashes at the end to prevent warning.
        stream.apply(add_hashes)
        
        # Setup model and put everything in memory
        self.model = SentenceTransformer(model_name)
        self.examples = list(stream)
        out = self.model.encode(["Test text right here."])
        self.index = Index(space="cosine", dim=out.shape[1])

        # If path is given, load from disk otherwise assume start from scratch
        if not index_path:
            self.index.init_index(max_elements=len(self.examples))
        else:
            self.index.load_index(str(index_path), max_elements=len(self.examples))
            log(f"RECIPE: Loaded index from {index_path}")
    
    def build_index(self, setting: Literal["text", "image"] = "text") -> "ApproximateIndex":
        # Index everything, progbar and save
        log(f"INDEX: About to build index with {setting=}.")
        iter_examples = tqdm(self.examples, desc="indexing")
        for batch in batched(iter_examples, n=256):
            if setting == "image":
                batch = [Image.open(ex['path']) for ex in batch]
            embeddings = self.model.encode(batch)
            self.index.add_items(embeddings)
        log(f"INDEX: Indexed {len(self.examples)} examples.")
        return self

    def store_index(self, path: Path):
        self.index.save_index(str(path))
        log(f"INDEX: Index file stored at {path}.")
    
    def new_stream(self, query:str, n:int=100):
        log(f"INDEX: Creating new stream of {n} examples using {query=}.")
        embedding = self.model.encode([query])[0]
        items, distances = self.index.knn_query([embedding], k=n)
        for lab, dist in zip(items[0].tolist(), distances[0].tolist()):
            # Get the original example
            ex = self.examples[int(lab)]

            # Add some extra meta info
            ex['meta'] = ex.get("meta", {})
            ex['meta']['index'] = int(lab)
            ex['meta']['distance'] = float(dist)
            ex['meta']["query"] = query
            
            # Don't forget hashes
            yield set_hashes(ex)


def stream_reset_calback(index_obj: ApproximateIndex, n:int=100):
    def stream_reset(ctrl: Controller, *, query: str):
        new_stream = Stream.from_iterable(index_obj.new_stream(query, n=n))
        ctrl.reset_stream(new_stream, prepend_old_wrappers=True)
        return next(ctrl.stream)
    return stream_reset


def remove_images(examples):
    # Remove all data URIs before storing example in the database
    for eg in examples:
        if eg.get("image", "").startswith("data:"):
            del eg["image"]
        yield set_hashes(eg)
