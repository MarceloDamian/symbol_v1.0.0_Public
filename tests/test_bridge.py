"""The bridge loop, driven by a scripted fake client instead of the API."""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

pytest.importorskip("anthropic")
from symbol.bridge.solve import run_apl, solve  # noqa: E402

COINS = "C←⍉(4⍴2)⊤¯1+⍳16⋄C⌿⍨15=C+.×3 5 7 11"


def _reply(*calls):
    content = [NS(type="tool_use", id="t%d" % i, name=name, input=inp) for i, (name, inp) in enumerate(calls)]
    return NS(stop_reason="tool_use", stop_details=None, content=content)


class FakeClient:
    """Answers each create() with the next scripted reply, recording what it was sent."""

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kwargs):
        self.sent.append(kwargs["messages"][-1])
        return self.replies.pop(0)


def test_run_apl_reports_results_and_errors():
    assert run_apl(COINS) == (True, "1 4⍴1 1 1 0")
    ok, text = run_apl("1 2 3[4]")
    assert not ok and text.startswith("error:")


def test_solve_tests_a_draft_then_submits_a_verified_program():
    client = FakeClient([
        _reply(("run_apl", {"program": "1 2 3[4]"})),  # a failing draft: error goes back as a tool result
        _reply(("submit", {"program": COINS, "answer": "Take 3, 5 and 7."})),
    ])
    found = solve("Which of 3 5 7 11 sum to 15?", client, verbose=False)
    assert found == {"program": COINS, "result": "1 4⍴1 1 1 0", "answer": "Take 3, 5 and 7."}
    error_result = client.sent[1]["content"][0]
    assert error_result["is_error"] and "error" in error_result["content"]


def test_solve_rejects_a_submit_that_does_not_run():
    client = FakeClient([
        _reply(("submit", {"program": "⍳", "answer": "?"})),
        _reply(("submit", {"program": "+/⍳4", "answer": "Ten."})),
    ])
    assert solve("sum 1..4", client, verbose=False)["result"] == "10"
