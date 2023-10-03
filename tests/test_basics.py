import srsly 
from pathlib import Path 
from prodigy_ann import index, fetch


def test_basics(tmpdir):
    examples_path = Path("tests/datasets/new-dataset.jsonl")
    index_path = tmpdir / "new-dataset.index"
    fetch_path = tmpdir / "fetched.jsonl"
    index(examples_path, index_path)
    fetch(examples_path, index_path, fetch_path, query="benchmarks")
    
    fetched_examples = list(srsly.read_jsonl(fetch_path))
    for ex in fetched_examples:
        assert ex['meta']['query'] == 'benchmarks'
