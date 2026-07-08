from tde.parsing.normalize import count_numeric_tokens, normalize
from tde.sources.base import EventRef, RawDoc


def test_sections_and_roles_from_components(doc_q1):
    sentences = normalize(doc_q1)
    assert sentences, "fixture should produce sentences"
    assert all(s.text.strip() for s in sentences), "no empty-text sentences"
    assert all(s.section in ("prepared", "qa") for s in sentences)

    prepared = [s for s in sentences if s.section == "prepared"]
    qa = [s for s in sentences if s.section == "qa"]
    assert prepared and qa

    roles = {(s.speaker, s.speaker_role) for s in sentences}
    assert ("Operator", "operator") in roles
    assert ("Pat Chief", "exec") in roles
    assert ("Alex Analyst", "analyst") in roles

    # idx strictly ordered
    assert [s.idx for s in sentences] == list(range(len(sentences)))
    # speakers all tagged in structured payloads
    assert all(s.speaker for s in sentences)


def test_numeric_tokens():
    # 10.2 | billion | 12 | %
    assert count_numeric_tokens("Revenue was 10.2 billion dollars, up 12% year over year.") == 4
    assert count_numeric_tokens("no numbers here") == 0
    # $ | 5 | million | 30 | bps
    assert count_numeric_tokens("$5 million and 30 bps") == 5


def test_raw_text_fallback_split():
    raw = (
        "Pat Chief: Good afternoon. Revenue grew nicely this quarter.\n"
        "Question-and-Answer Session\n"
        "Operator: Our first question please.\n"
        "Randy Researcher: What drove the margin upside this quarter?\n"
        "Pat Chief: Mostly mix and utilization gains.\n"
    )
    doc = RawDoc(event=EventRef("e", "2025-01-01"), raw_text=raw)
    sentences = normalize(doc)
    by_speaker = {}
    for s in sentences:  # first sentence per speaker
        by_speaker.setdefault(s.speaker, s)
    assert by_speaker["Pat Chief"].section == "prepared"
    assert by_speaker["Randy Researcher"].section == "qa"
    assert by_speaker["Randy Researcher"].speaker_role == "analyst"
    assert by_speaker["Operator"].speaker_role == "operator"
    # exec speaking in Q&A keeps exec role (seen in prepared)
    qa_pat = [s for s in sentences if s.speaker == "Pat Chief" and s.section == "qa"]
    assert qa_pat and qa_pat[0].speaker_role == "exec"
