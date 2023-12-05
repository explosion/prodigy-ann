import pytest 
from playwright.sync_api import expect

from .e2e_util import prodigy_playwright


base_calls = [
    "textcat.ann.manual xxx",
    "ner.ann.manual xxx blank:en",
    "spans.ann.manual xxx blank:en"
]

@pytest.mark.parametrize("query", ["benchmark", "corpus"])
@pytest.mark.parametrize("base_call", base_calls)
@pytest.mark.e2e
def test_basic_interactions(query, base_call):
    """Ensure that we check e2e that the query appears when we reset the stream."""
    extra_settings = "tests/datasets/new-dataset.jsonl tests/datasets/new-dataset.index --query download --allow-reset --n 20 --labels foo,bar,buz"
    with prodigy_playwright(f"{base_call} {extra_settings}") as (ctx, page):
        # Reset the stream
        page.get_by_text("Reset stream?").click()
        page.get_by_label("New query for ANN:").click()
        page.get_by_label("New query for ANN:").fill(query)
        page.get_by_role("button", name="Refresh Stream").click()
        page.get_by_text("Reset stream?").click()
        
        # Hit accept a few times, making sure that the query appears
        for _ in range(10):
            page.get_by_label("accept (a)").click()
            elem = page.locator(".prodigy-content").first
            expect(elem).to_contain_text(query)
