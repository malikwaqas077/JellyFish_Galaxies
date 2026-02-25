"""
TNG REST API helpers — all HTTP calls go through here.
"""

import time
import requests
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import API_KEY, REQUEST_TIMEOUT, DOWNLOAD_RETRIES

HEADERS = {"api-key": API_KEY}


def get(url, params=None, stream=False):
    """GET with retries and polite error messages."""
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS,
                             timeout=REQUEST_TIMEOUT, stream=stream)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None          # resource genuinely missing
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except requests.exceptions.RequestException:
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    return None


def get_json(url, params=None):
    r = get(url, params=params)
    if r is None:
        return None
    return r.json()


def download_file(url, dest_path, params=None):
    """Stream-download a binary file (e.g. HDF5 cutout)."""
    r = get(url, params=params, stream=True)
    if r is None:
        return False
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            f.write(chunk)
    return True


def paginate(url, params=None):
    """Follow TNG API pagination and collect all results."""
    params = params or {}
    results = []
    while url:
        data = get_json(url, params=params)
        if data is None:
            break
        results.extend(data.get("results", []))
        url = data.get("next")
        params = {}          # next URL already has params encoded
    return results
