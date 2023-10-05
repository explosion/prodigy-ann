import spacy 
import srsly 
from pathlib import Path 
from prodigy_ann import index, fetch, textcat_ann_manual, ner_ann_manual, spans_ann_manual


def test_basics(tmpdir):
    examples_path = Path("tests/datasets/new-dataset.jsonl")
    index_path = tmpdir / "new-dataset.index"
    fetch_path = tmpdir / "fetched.jsonl"
    query = "benchmarks"

    # Ensure fetch works as expected
    index(examples_path, index_path)
    fetch(examples_path, index_path, fetch_path, query="benchmarks")
    
    fetched_examples = list(srsly.read_jsonl(fetch_path))
    for ex in fetched_examples:
        assert ex['meta']['query'] == 'benchmarks'

    # Also ensure the helpers do not break
    nlp = spacy.blank("en")
    textcat_ann_manual("xxx", examples_path, index_path, labels="foo,bar", query=query)
    ner_ann_manual("xxx", nlp, examples_path, index_path, labels="foo,bar", query=query)
    spans_ann_manual("xxx", nlp, examples_path, index_path, labels="foo,bar", query=query)

