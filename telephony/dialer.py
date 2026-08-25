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
import requests


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