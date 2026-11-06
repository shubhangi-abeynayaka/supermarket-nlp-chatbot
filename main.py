"""
Supermarket Assistant Chatbot  (CO3251 - Natural Language Processing, Assignment 2)

Pipeline:
    1. NLP extraction   - spaCy noun chunks + noun tokens + lemmas
                          (falls back to word n-grams if spaCy is unavailable)
    2. Fuzzy matching   - RapidFuzz similarity against a canonical product index
                          (falls back to difflib if RapidFuzz is unavailable)
    3. Resolution       - longest phrase first, word-level de-duplication,
                          canonical product names
    4. Output           - shelf list in the terminal + optional printable .txt file
"""
import re
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional dependency: RapidFuzz (fast, typo-tolerant matching)
try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Optional dependency: spaCy (POS tagging, noun chunks, lemmatization)
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
except ImportError:
    nlp = None


# ----------------------------------------------------------------------------
# 1. DATABASE & SETTINGS
# ----------------------------------------------------------------------------

# search phrase -> (canonical product name, shelf location)
SHELF_DATABASE: Dict[str, Tuple[str, str]] = {
    # Dairy & Chilled
    "milk": ("Milk", "Shelf A1"),
    "whole milk": ("Milk", "Shelf A1"),
    "cheese": ("Cheese", "Shelf A1"),
    "butter": ("Butter", "Shelf A1"),
    "yogurt": ("Yogurt", "Shelf A2"),
    "ice cream": ("Ice Cream", "Shelf A2"),
    # Fresh Produce
    "apple": ("Apples", "Shelf B1"),
    "apples": ("Apples", "Shelf B1"),
    "fresh apples": ("Apples", "Shelf B1"),
    "banana": ("Bananas", "Shelf B1"),
    "bananas": ("Bananas", "Shelf B1"),
    "ripe bananas": ("Bananas", "Shelf B1"),
    "carrot": ("Carrots", "Shelf B2"),
    "carrots": ("Carrots", "Shelf B2"),
    "tomato": ("Tomatoes", "Shelf B2"),
    "tomatoes": ("Tomatoes", "Shelf B2"),
    # Bakery & Pantry
    "bread": ("Bread", "Shelf C1"),
    "cereal": ("Cereal", "Shelf C1"),
    "pasta": ("Pasta", "Shelf C2"),
    "rice": ("Rice", "Shelf C2"),
    "coffee": ("Coffee", "Shelf C2"),
    "instant coffee": ("Instant Coffee", "Shelf C2"),
    # Household & Care
    "detergent": ("Detergent", "Shelf D1"),
    "toilet paper": ("Toilet Paper", "Shelf D2"),
}
DB_KEYS = list(SHELF_DATABASE)

CONFIDENCE_THRESHOLD = 75   # minimum similarity (0-100) for normal-length words
SHORT_WORD_LEN = 5          # words this short or shorter need a stricter score,
SHORT_WORD_THRESHOLD = 90   # otherwise "price" would be read as "rice"

# Quantity / filler words that are never products
FILLER = {
    "bag", "bags", "packet", "packets", "pack", "packs", "box", "boxes",
    "bottle", "bottles", "carton", "kg", "some", "need", "want", "buy",
    "grab", "get", "please", "thing", "things", "item", "items", "list",
    "lot", "bit", "piece", "pieces", "today",
}


# ----------------------------------------------------------------------------
# 2. NLP EXTRACTION
# ----------------------------------------------------------------------------

# Articles, prepositions, pronouns and request verbs that can never be part of a
# product name. They are trimmed from the edges of every candidate phrase.
STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "with", "my", "me",
    "i", "is", "are", "where", "can", "could", "do", "does", "here", "there",
    "looking", "find", "show", "tell", "much", "many", "any", "more", "like",
}
EDGE_WORDS = FILLER | STOPWORDS


def _clean(text: str) -> str:
    """Lowercase and keep letters/spaces only."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", text.lower())).strip()


def _trim_edges(phrase: str) -> str:
    """Drop stop/filler words from both ends: 'some instant cofee' -> 'instant cofee'."""
    words = phrase.split()
    while words and words[0] in EDGE_WORDS:
        words.pop(0)
    while words and words[-1] in EDGE_WORDS:
        words.pop()
    return " ".join(words)


def ngrams(text: str, max_n: int = 3) -> List[str]:
    """
    Word n-grams (longest first). Used when spaCy is not available.
    n-grams are built inside one list segment only, never across a comma or
    'and', so "fresh bananas, apples" cannot become "fresh bananas apples".
    """
    segments = re.split(r"[,;.!?]|\b(?:and|or|plus|also)\b", text.lower())
    grams = []
    for segment in segments:
        words = re.findall(r"[a-z]+", segment)
        for n in range(min(max_n, len(words)), 0, -1):
            grams.extend(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    return grams


def extract_candidate_items(text: str) -> List[str]:
    """Return candidate item phrases, longest first, so full names win."""
    if nlp:
        candidates = []
        doc = nlp(text)
        # Multi-word noun chunks, without determiners/pronouns ("some instant cofee")
        for chunk in doc.noun_chunks:
            phrase = _clean(" ".join(
                t.text for t in chunk if t.pos_ not in {"DET", "PRON"}
            ))
            if phrase:
                candidates.append(phrase)
        # Single nouns and their lemmas ("tomatoes" -> "tomato")
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                candidates.append(_clean(token.text))
                candidates.append(_clean(token.lemma_))
    else:
        candidates = ngrams(text)

    trimmed = (_trim_edges(c) for c in candidates)
    unique = [c for c in dict.fromkeys(trimmed) if c]
    return sorted(unique, key=len, reverse=True)


# ----------------------------------------------------------------------------
# 3. MATCHING & DE-DUPLICATION
# ----------------------------------------------------------------------------

def required_score(candidate: str) -> float:
    """Short words need a stricter match to avoid false positives."""
    return SHORT_WORD_THRESHOLD if len(candidate) <= SHORT_WORD_LEN else CONFIDENCE_THRESHOLD


def best_match(candidate: str) -> Optional[Tuple[str, float]]:
    """Closest database key for a candidate, or None if below the threshold."""
    key, score = None, 0.0
    if HAS_RAPIDFUZZ:
        hit = process.extractOne(candidate, DB_KEYS, scorer=fuzz.ratio)
        if hit:
            key, score = hit[0], float(hit[1])
    else:
        for db_key in DB_KEYS:
            s = SequenceMatcher(None, candidate, db_key).ratio() * 100
            if s > score:
                key, score = db_key, s
    if key is not None and score >= required_score(candidate):
        return key, score
    return None


def match_items(candidates: List[str]) -> List[Tuple[str, str, str, int]]:
    """
    Map candidates to shelves.
    Returns (canonical name, shelf, what the user typed, confidence %).
    """
    results = []
    seen_products = set()
    used_words = set()   # words already consumed by a longer matched phrase

    for candidate in candidates:
        words = set(candidate.split())
        if not words or words <= FILLER or words & used_words:
            continue
        hit = best_match(candidate)
        if hit is None:
            continue
        key, score = hit
        canonical, shelf = SHELF_DATABASE[key]
        used_words |= words
        if canonical not in seen_products:
            seen_products.add(canonical)
            results.append((canonical, shelf, candidate, round(score)))
    return results


# ----------------------------------------------------------------------------
# 4. OUTPUT
# ----------------------------------------------------------------------------

def format_results(results: List[Tuple[str, str, str, int]]) -> List[str]:
    """Shelf lines, sorted by shelf so the customer can walk the store in order."""
    lines = []
    for canonical, shelf, typed, score in sorted(results, key=lambda r: r[1]):
        note = "exact" if typed == canonical.lower() else f"you typed '{typed}', {score}% match"
        lines.append(f" - {canonical:<16} -> {shelf:<9} ({note})")
    return lines


def save_receipt(results: List[Tuple[str, str, str, int]], path: str = "shelf_list.txt") -> Path:
    """Write a printable shelf list and return its absolute path."""
    out = Path(path)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"SHELF LIST - {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write("-" * 36 + "\n")
        for canonical, shelf, _, _ in sorted(results, key=lambda r: r[1]):
            f.write(f"{canonical:<18} {shelf}\n")
    return out.resolve()


# ----------------------------------------------------------------------------
# 5. TERMINAL INTERFACE
# ----------------------------------------------------------------------------

def main() -> None:
    bar = "=" * 62
    print(bar)
    print("            SUPERMARKET ASSISTANT CHATBOT")
    print(bar)
    if not nlp:
        print("[!] spaCy model not found - using basic n-gram extraction.")
        print("    Run: python -m spacy download en_core_web_sm")
    if not HAS_RAPIDFUZZ:
        print("[!] RapidFuzz not found - using difflib for matching.")
    print("Tell me what you need and I will find the shelves.")
    print("Example: 'I need some whole milk, a bag of carrots, and instant coffee'")
    print("Type 'quit' to exit.")

    while True:
        try:
            text = input("\nCustomer Request > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not text:
            print("Please type the items you are looking for.")
            continue

        results = match_items(extract_candidate_items(text))

        print("\n" + bar)
        print("                      SHELF LOCATION")
        print(bar)
        if not results:
            print(" Sorry, I could not find any of those items in our store.")
            print(bar)
            continue
        print("\n".join(format_results(results)))
        print(bar)

        try:
            answer = input("Save a printable shelf list? (y/n) > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"
        if answer.startswith("y"):
            print(f"Saved to {save_receipt(results)}")


if __name__ == "__main__":
    main()