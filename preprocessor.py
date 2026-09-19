"""
Text Preprocessing Pipeline for Email and SMS Spam Detection.
Includes lowercase conversion, punctuation removal, stopword removal,
and stemming/lemmatization.
"""

from pathlib import Path
import re
import string
import nltk

# Configure local project nltk_data directory if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_NLTK_DATA = PROJECT_ROOT / "nltk_data"
if LOCAL_NLTK_DATA.exists():
    nltk.data.path.insert(0, str(LOCAL_NLTK_DATA))

# Standard English stopwords fallback
FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "u", "ur", "r", "c", "ok", "im", "dont"
}

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = FALLBACK_STOPWORDS

try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    STEMMER = PorterStemmer()
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    STEMMER = None
    LEMMATIZER = None


def clean_text(text: str) -> str:
    """
    Applies initial regex cleaning to raw email/message text:
    - Strips URLs
    - Strips email addresses
    - Replaces HTML tags
    - Retains only alphabetic characters and spaces
    - Collapses consecutive whitespace
    """
    if not isinstance(text, str):
        return ""

    # Lowercase conversion
    text = text.lower()

    # Remove URLs (http/https/www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove HTML tags if present
    text = re.sub(r"<.*?>", " ", text)

    # Replace non-alphabetic characters (punctuations, numbers, symbols) with space
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    use_stemming: bool = True,
    use_lemmatization: bool = False
) -> str:
    """
    Full text preprocessing pipeline:
    1. Lowercase conversion & regex cleanup
    2. Tokenization into words
    3. Stopword removal
    4. Stemming (PorterStemmer) or Lemmatization (WordNetLemmatizer)

    Parameters
    ----------
    text : str
        The input text string.
    remove_stopwords : bool, default True
        Whether to filter out English stopwords.
    use_stemming : bool, default True
        Whether to apply Porter stemming to tokens.
    use_lemmatization : bool, default False
        Whether to apply WordNet lemmatization to tokens.

    Returns
    -------
    str
        Processed, normalized text string.
    """
    cleaned = clean_text(text)
    words = cleaned.split()

    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 1]

    if use_stemming and STEMMER:
        words = [STEMMER.stem(w) for w in words]
    elif use_lemmatization and LEMMATIZER:
        words = [LEMMATIZER.lemmatize(w) for w in words]

    return " ".join(words)


if __name__ == "__main__":
    sample_ham = "Hey! Are you coming to the party tonight? Call me at 555-0199 or visit https://example.com!"
    sample_spam = "WINNER!! As a valued customer, you have won a £900 prize! Claim now at http://win-big.com or text PRIZE to 8888!"

    print("--- Preprocessing Demo ---")
    print("Original Ham :", sample_ham)
    print("Cleaned Ham  :", preprocess_text(sample_ham))
    print("\nOriginal Spam:", sample_spam)
    print("Cleaned Spam :", preprocess_text(sample_spam))
