import srsly 
from pathlib import Path 
from prodigy_ann.image import image_fetch, image_ann_manual, image_index


def test_basics(tmpdir):
    examples_path = Path("tests/datasets/images")
    index_path = tmpdir / "new-dataset.index"
    fetch_path = tmpdir / "out.jsonl"
    query = "laptop"

    # Ensure fetch works as expected
    image_index(examples_path, index_path)
    image_fetch(examples_path, index_path, fetch_path, query="laptop", n=4)
    
    fetched_examples = list(srsly.read_jsonl(fetch_path))
    for ex in fetched_examples:
        assert ex['meta']['query'] == query
        assert query in ex['path']

    # Also ensure the helpers do not break, this is a smoke-check
    out = image_ann_manual("xxx", examples_path, index_path, labels="laptop", query=query, n=4)
    for ex in out['stream']:
        assert ex['meta']['query'] == query
        assert query in ex['path']
