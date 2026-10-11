"""Turn requests like "Generate me 200 dragon pictures" into (count, subject)."""

import re
from dataclasses import dataclass

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "dozen": 12,
}

FILLER = {
    "generate", "make", "create", "draw", "give", "show", "render", "produce",
    "me", "us", "please", "plz", "pls", "can", "you", "could", "would", "i",
    "want", "need", "some", "of", "a", "an", "the", "pictures", "picture",
    "images", "image", "photos", "photo", "pics", "pic", "drawings", "drawing",
    "art", "artworks", "versions", "different", "new", "for", "with", "ty",
    "tysm", "thanks", "thank",
}


@dataclass
class ImageRequest:
    count: int
    subject: str


def parse_request(text: str, default_count: int = 1) -> ImageRequest:
    words = re.findall(r"[a-zA-Z]+|\d+", text.lower())

    count = None
    remaining = []
    for word in words:
        if count is None and word.isdigit():
            count = int(word)
        elif count is None and word in NUMBER_WORDS and word not in ("a", "an"):
            count = NUMBER_WORDS[word]
        else:
            remaining.append(word)

    subject_words = [w for w in remaining if w not in FILLER]
    if not subject_words:
        raise ValueError(f"Couldn't find what to draw in: {text!r}")

    subject_words[-1] = singular(subject_words[-1])  # "dragons" -> "dragon"
    return ImageRequest(count=count or default_count, subject=" ".join(subject_words))


def singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def match_class(subject: str, class_names: list[str]) -> str | None:
    """Find the trained class that best matches a free-text subject."""
    norm = lambda s: " ".join(singular(w) for w in s.lower().replace("_", " ").split())
    target = norm(subject)
    lookup = {norm(name): name for name in class_names}

    if target in lookup:
        return lookup[target]
    # "red dragon" still matches a "dragon" class, "dragons" matches "dragon".
    target_words = set(target.split())
    for key, name in lookup.items():
        if set(key.split()) <= target_words:
            return name
    return None
