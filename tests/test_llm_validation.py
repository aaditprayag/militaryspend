from tde.synth.llm import _extract_json, validate_quotes


def test_extract_json_plain_and_fenced():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('noise before {"a": 1} noise after') == {"a": 1}


def test_validate_quotes_drops_bad():
    sentences = [
        "Backlog grew again this quarter and visibility now extends beyond four quarters.",
    ]
    result = {
        "new_topics": [
            {
                "topic": "backlog",
                "evidence_quote": "visibility now extends beyond four quarters",
                "speaker": "Pat Chief",
            },
            {
                "topic": "fabricated",
                "evidence_quote": "we expect margins to collapse",
                "speaker": "Pat Chief",
            },
            {"topic": "too long", "evidence_quote": " ".join(["word"] * 30), "speaker": "X"},
        ],
        "flags": [{"severity": "low", "note": "n"}],
    }
    out = validate_quotes(result, sentences)
    assert len(out["new_topics"]) == 1
    assert out["new_topics"][0]["topic"] == "backlog"
    # missing keys are defaulted so templates never KeyError
    assert out["dropped_topics"] == []
    assert out["one_paragraph_read"] == ""
    assert out["flags"] == [{"severity": "low", "note": "n"}]
