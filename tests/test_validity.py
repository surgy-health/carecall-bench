"""The language check behind the paper's validity appendix (eval/validity.py)."""

from eval.validity import HINDI, is_hinglish, turns


def test_hinglish_caller_lines_are_detected():
    assert is_hinglish("Prawns khaane ke baad mere husband ke lips aur face sooj gaye hain")
    assert is_hinglish("Mujhe knee mein pain hai, Wednesday ko dikhana hai")
    assert is_hinglish("मुझे अपॉइंटमेंट चाहिए")  # Devanagari counts on its own


def test_english_lines_are_not_flagged():
    for line in (
        "I want to cancel my appointment on Friday, 20 November.",
        "Can you book me with the cardiologist the day after tomorrow?",
        "My papa had chest pain so I am calling to ask about the main doctor.",
    ):
        assert not is_hinglish(line), line


def test_word_list_leaves_out_common_english_words():
    assert not {"the", "to", "main", "do", "he", "so", "par", "papa", "beta"} & HINDI


def test_turns_drops_tokens_and_short_replies():
    sim = {"messages": [
        {"role": "user", "content": "###STOP###"},
        {"role": "user", "content": "Yes."},
        {"role": "user", "content": "I need an appointment for my knee please ###STOP###"},
        {"role": "assistant", "content": "Sure, let me check the slots for you."},
    ]}
    assert turns(sim, "user") == ["I need an appointment for my knee please"]
    assert turns(sim, "assistant") == ["Sure, let me check the slots for you."]
