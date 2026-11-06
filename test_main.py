"""Run with:  pytest -q"""
import pytest

import main


def products(text):
    """Set of canonical product names found in a sentence."""
    return {r[0] for r in main.match_items(main.extract_candidate_items(text))}


@pytest.mark.parametrize("text, expected", [
    # assignment example
    ("I want to buy apples, milk, and detergent.", {"Apples", "Milk", "Detergent"}),
    # guide example 1: compound phrase + typo + filler words
    ("I need to grab some whole milk, a bag of carrots, and some instant cofee "
     "for my morning routine", {"Milk", "Carrots", "Instant Coffee"}),
    # guide example 2: plurals
    ("Looking for fresh bananas, apples, and tomatoes", {"Bananas", "Apples", "Tomatoes"}),
    # typos
    ("where is the detergnt and yorgurt?", {"Detergent", "Yogurt"}),
    # multi-word items are not split into wrong single words
    ("ice cream and toilet paper", {"Ice Cream", "Toilet Paper"}),
    # duplicates collapse to one entry
    ("milk, whole milk and more milk", {"Milk"}),
])
def test_expected_items(text, expected):
    assert products(text) == expected


def test_out_of_store_items_are_rejected():
    assert products("Can I buy a laptop, shoes, and a bicycle here?") == set()


def test_short_lookalike_word_is_not_a_product():
    # "price" is 89% similar to "rice"; the short-word rule must reject it
    assert products("what is the price of milk") == {"Milk"}


def test_empty_input():
    assert main.match_items(main.extract_candidate_items("")) == []


def test_receipt_is_written(tmp_path):
    results = main.match_items(main.extract_candidate_items("milk and bread"))
    out = main.save_receipt(results, str(tmp_path / "list.txt"))
    text = out.read_text(encoding="utf-8")
    assert "Milk" in text and "Shelf A1" in text and "Bread" in text
