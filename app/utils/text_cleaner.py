import re


def clean_for_tts(text: str) -> str:
    # Remove markdown formatting
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)

    # Remove headers
    text = re.sub(r"#+\s*", "", text)

    # Remove bullet points
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)

    # Collapse multiple newlines
    text = re.sub(r"\n+", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()