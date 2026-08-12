import json
import re
from pathlib import Path


REQUIREMENT_KEYWORDS = {
    "pvc cards": "PVC cards",
    "pvc card": "PVC cards",
    "rfid": "RFID",
    "mifare": "MIFARE cards",
    "nfc": "NFC cards",
    "id card printer": "ID card printers",
    "id card printers": "ID card printers",
    "printer": "ID card printers",
    "printers": "ID card printers",
    "ribbons": "Printer ribbons",
    "ribbon": "Printer ribbons",
    "consumables": "Printer consumables",
    "lanyard": "Lanyards",
    "lanyards": "Lanyards",
    "access control": "Access control",
    "biometric": "Biometric solutions",
    "attendance": "Attendance solutions",
    "card designing": "Card designing",
    "card design": "Card designing",
    "software": "ID card software",
}

BRANDS = {
    "hid": "HID",
    "hid fargo": "HID Fargo",
    "zebra": "Zebra",
    "evolis": "Evolis",
    "datacard": "DataCard",
    "idp": "IDP",
}

INDIAN_LOCATIONS = {
    "delhi",
    "noida",
    "bangalore",
    "bengaluru",
    "pune",
    "gujarat",
    "gujrat",
    "arunachal",
    "arunachal pradesh",
    "mumbai",
    "chennai",
    "hyderabad",
    "kolkata",
    "jaipur",
    "lucknow",
    "ahmedabad",
}


def load_transcript(file_path):
    messages = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            messages.append(json.loads(line))

    return messages


def extract_name(messages):
    """
    Looks for:
    Spark: May I get your name?
    User: arjun
    """

    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        text = message["content"].lower()

        if "get your name" in text or "what is your name" in text:
            if i + 1 < len(messages):
                answer = messages[i + 1]

                if answer["role"] == "user":
                    name = answer["content"].strip()

                    # Avoid treating a long sentence as a name
                    if len(name.split()) <= 5:
                        return name.title()

    return None


def extract_company(messages):
    """
    Looks for:
    Spark: Which organization are you calling from?
    User: SBC Hotel
    """

    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        text = message["content"].lower()

        organization_question = (
            "which organization" in text
            or "what organization" in text
            or "name of your organization" in text
            or "organization are you calling from" in text
            or "company are you calling from" in text
        )

        if organization_question:
            if i + 1 < len(messages):
                answer = messages[i + 1]

                if answer["role"] == "user":
                    company = answer["content"].strip()

                    if len(company.split()) <= 8:
                        return company

    return None


def extract_callback_time(messages):
    """
    Looks for:
    Spark: What time would you prefer for the callback?
    User: around 5pm on friday
    """

    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        text = message["content"].lower()

        if (
            "what time" in text
            and "callback" in text
        ):
            if i + 1 < len(messages):
                answer = messages[i + 1]

                if answer["role"] == "user":
                    return answer["content"].strip()

    return None


def extract_requirements(messages):
    requirements = []

    requirement_triggers = [
        "need",
        "needs",
        "needing",
        "looking for",
        "looking to get",
        "require",
        "requires",
        "want",
        "wants",
        "would like",
        "we use",
        "we currently use",
        "we have",
        "we might need",
        "we may need",
        "do you have",
    ]

    for message in messages:
        if message["role"] != "user":
            continue

        text = message["content"].strip()
        lower = text.lower()

        if any(trigger in lower for trigger in requirement_triggers):
            requirements.append(text)

    return requirements

def extract_brands(messages):
    brands = set()

    for message in messages:
        if message["role"] != "user":
            continue

        text = message["content"].lower()

        for keyword, value in BRANDS.items():
            if keyword in text:
                brands.add(value)

    return sorted(brands)


def extract_location(messages):
    locations = []

    location_triggers = [
        "i am from",
        "i'm from",
        "we are from",
        "we're from",
        "based in",
        "located in",
        "located at",
        "our office is in",
        "office is in",
        "company is in",
        "we operate in",
        "deliver to",
        "delivery to",
        "ship to",
        "shipping to",
        "service in",
        "support in",
    ]

    for message in messages:
        if message["role"] != "user":
            continue

        text = message["content"].strip()
        lower = text.lower()

        for trigger in location_triggers:
            if trigger in lower:
                # Keep the customer's original sentence.
                locations.append(text)
                break

    return locations

def extract_quantity(messages):
    """
    Finds things like:
    500 cards
    1000 cards
    around 500
    """

    for message in messages:
        if message["role"] != "user":
            continue

        text = message["content"].lower()

        match = re.search(
            r"\b(\d[\d,]*)\b\s*(?:cards?|units?|pieces?|pcs?)",
            text
        )

        if match:
            return int(match.group(1).replace(",", ""))

    return None


def extract_status(messages):
    text = " ".join(
        message["content"].lower()
        for message in messages
    )

    if "our specialist will contact you" in text:
        return "callback_scheduled"

    if any(
        phrase in text
        for phrase in [
            "not interested",
            "no thanks",
            "don't want",
            "do not want",
        ]
    ):
        return "rejected"

    return "completed"


def parse_transcript(file_path):
    messages = load_transcript(file_path)

    return {
        "name": extract_name(messages),
        "company": extract_company(messages),
        "requirements": extract_requirements(messages),
        "quantity": extract_quantity(messages),
        "location": extract_location(messages),
        "brands": extract_brands(messages),
        "callback_time": extract_callback_time(messages),
        "status": extract_status(messages),
        "transcript_file": str(file_path),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage:")
        print("python -m app.parser transcripts/your_file.json")
        raise SystemExit(1)

    result = parse_transcript(sys.argv[1])

    print(json.dumps(result, indent=4, ensure_ascii=False))