# Supermarket Assistant Chatbot — NLP Shelf Locator

A lightweight CLI chatbot that parses free-text grocery requests into store shelf locations using natural language processing (NLTK) and fuzzy string matching.

## How It Works
1. **Text Extraction:** Tokenizes, filters Part-of-Speech (POS) tags, and lemmatizes user queries using NLTK to isolate candidate product nouns from conversational filler.
2. **Fuzzy Catalogue Matching:** Matches extracted terms against a 17-product store catalogue using `RapidFuzz` similarity scoring. Applies a 75% similarity threshold for standard terms and a stricter 90% threshold for short tokens to suppress false positives, prioritizing longest-phrase matches first.
3. **Shelf List Output:** Formats confirmed items into an itemized shelf locator list and automatically exports the output to `shelf_list.txt`.

## Setup Instructions

### Prerequisites
- Python 3.10+

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shubhangi-abeynayaka/supermarket-nlp-chatbot.git
   cd supermarket-nlp-chatbot
   ```

2. Install dependencies:
   ```bash
   py -m pip install -r requirements.txt
   ```

## Usage & Example

Run the chatbot script:
```bash
py main.py
```

### Sample Session
```text
Input:  "Hi, where can I find some fresh apples, 2 cartons of whole milk, and laundry detergent?"

Output:
========================================
         SUPERMARKET SHELF LOCATOR      
========================================
[+] Found: Apple               -> Aisle 2 (Produce)
[+] Found: Whole Milk          -> Aisle 5 (Dairy)
[+] Found: Laundry Detergent   -> Aisle 9 (Household)
----------------------------------------
Exported shelf list to shelf_list.txt
```

## Running Tests

Execute the unit test suite with `pytest`:
```bash
py -m pytest -q
```

## Repository Structure
```text
supermarket-nlp-chatbot/
├── main.py
├── test_main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations
- **Catalogue Size:** Evaluated against a 17-product test catalogue.
- **Language:** English input support only.
- **Short Token Sensitivity:** Words under 4 characters require near-exact spelling due to elevated threshold boundaries.
