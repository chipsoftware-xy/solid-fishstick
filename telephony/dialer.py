"""
Places an outbound call via Exotel — the Flow + Voicebot applet
method, since that's the one that actually worked (not the direct
streamurl/AgentStream method, which stayed blocked behind account
enablement the whole time we tested it).

The telephony server itself (two_way_test.py) isn't started from here
— its WebSocket URL is configured once, directly in the Exotel Flow's
Voicebot applet, in their dashboard. This module only fires the API
request that makes the phone actually ring; the telephony server just
needs to already be running and reachable at whatever URL that applet
points to.
"""
import re
import yaml
from pathlib import Path

import requests

SETTINGS_PATH = Path("config/settings.yml")
EXOTEL_FIELDS = ["api_key", "api_token", "account_sid", "exophone", "flow_id"]


def normalize_phone(val):
    """
    Normalizes a phone number into the +91XXXXXXXXXX format (E.164) —
    strips pandas' trailing '.0' on number-formatted cells, strips a
    leading 0 (common domestic-style habit), and adds the +91 prefix.
    Shared by both the Call Queue upload (server.py) and the telephony
    server (two_way_test.py) — Exotel reports the caller's number in
    plain domestic format in its 'start' event (e.g. '09650707992'),
    while Call Queue stores E.164 ('+919650707992'). Without both
    sides going through this same normalization, marking a call done/
    failed by phone number silently never matches anything — confirmed
    real bug, present in every test throughout this whole telephony
    phase.
    """
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]

    # Strip anything that isn't a digit or a leading +, in case a
    # space or dash snuck in.
    s = re.sub(r"[^\d+]", "", s)

    if s.startswith("+"):
        return s  # already fully formatted — leave it alone

    if s.startswith("91") and len(s) == 12:
        return "+" + s  # has the country code, just missing the +

    if s.startswith("0"):
        s = s[1:]  # strip a leading domestic-style 0

    if len(s) == 10:
        return "+91" + s

    # Doesn't match a recognized shape — leave as-is rather than guess
    # further.
    return s


def get_exotel_settings():
    if not SETTINGS_PATH.exists():
        return {}
    with open(SETTINGS_PATH) as f:
        data = yaml.safe_load(f) or {}
    return data.get("exotel", {})


def set_exotel_settings(values):
    with open(SETTINGS_PATH) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("exotel", {}).update(values)
    with open(SETTINGS_PATH, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


class ExotelConfigError(Exception):
    """Raised when required Exotel credentials are missing."""
    pass


def place_call(phone_number, exotel_settings):
    """
    exotel_settings is a dict with: api_key, api_token, account_sid,
    exophone, flow_id — all required. Raises ExotelConfigError if any
    are missing, or requests.HTTPError if Exotel itself rejects the
    request.
    """
    required = ["api_key", "api_token", "account_sid", "exophone", "flow_id"]
    missing = [k for k in required if not exotel_settings.get(k)]
    if missing:
        raise ExotelConfigError(
            f"Missing Exotel setting(s): {', '.join(missing)} — "
            f"configure these under Settings first."
        )

    api_key = exotel_settings["api_key"]
    api_token = exotel_settings["api_token"]
    account_sid = exotel_settings["account_sid"]

    url = (
        f"https://{api_key}:{api_token}@api.in.exotel.com/v1/Accounts/"
        f"{account_sid}/Calls/connect"
    )
    data = {
        "From": phone_number,
        "CallerId": exotel_settings["exophone"],
        "Url": (
            f"http://my.exotel.com/{account_sid}/exoml/start_voice/"
            f"{exotel_settings['flow_id']}"
        ),
    }

    response = requests.post(url, data=data, timeout=15)
    response.raise_for_status()
    return response