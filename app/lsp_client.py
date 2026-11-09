# Only needed for the LSP extension.
# You can ignore this file until you reach that extension.
"""Course-supplied LSP transport client. Document state belongs in your agent."""

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict


class LSPClient:

    @classmethod
    def from_environment(cls):
        if "CODECRAFTERS_LSP_URL" not in os.environ:
            return None
        return cls()

    def __init__(self):
        self.url = os.environ["CODECRAFTERS_LSP_URL"]
        self.extension_to_language = self.call("configuration", {})["extensionToLanguage"]

    def call(self, action: str, payload: Dict[str, Any]):
        print(f"Calling LSP action '{action}' with payload: {payload}", file=sys.stderr)
        request = urllib.request.Request(
            self.url + "/" + action,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode("utf-8")) from error

    def request(self, method: str, params: Dict[str, Any]):
        return self.call("request", {"method": method, "params": params})

    def notify(self, method: str, params: Dict[str, Any]):
        return self.call("notify", {"method": method, "params": params})

    def notifications(self, after: int = 0, timeout_ms: int = 0):
        return self.call("notifications", {"after": after, "timeoutMs": timeout_ms})
