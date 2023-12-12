import pytest 
import srsly 
from pathlib import Path 
from prodigy_ann.text import text_index, text_fetch, textcat_ann_manual, ner_ann_manual, spans_ann_manual


def test_basics(tmpdir):
    examples_path = Path("tests/datasets/new-dataset.jsonl")
    index_path = tmpdir / "new-dataset.index"
    fetch_path = tmpdir / "fetched.jsonl"
    query = "benchmarks"

    # Ensure fetch works as expected
    text_index(examples_path, index_path)
    text_fetch(examples_path, index_path, fetch_path, query=query)

    # Can't fetch more than the examples we have
    with pytest.raises(SystemExit):
        text_fetch(examples_path, index_path, fetch_path, query=query, n=100_000)
    
    fetched_examples = list(srsly.read_jsonl(fetch_path))
    for ex in fetched_examples:
        assert ex['meta']['query'] == query

    # Also ensure the helpers do not break
    out = textcat_ann_manual("xxx", examples_path, index_path, labels="foo,bar", query=query)
    assert isinstance(out, dict)
    assert next(out['stream'])

    out = ner_ann_manual("xxx", "blank:en", examples_path, index_path, labels="foo,bar", query=query)
    assert isinstance(out, dict)
    assert next(out['stream'])

    out = spans_ann_manual("xxx", "blank:en", examples_path, index_path, labels="foo,bar", query=query)
    assert isinstance(out, dict)
    assert next(out['stream'])

