from google.genai import types
from loom.compaction import compact, KEEP_LAST
from unittest.mock import MagicMock

def test_compact_too_short():
    client = MagicMock()
    messages = [
        types.Content(role="user", parts=[types.Part(text="a")]),
        types.Content(role="model", parts=[types.Part(text="b")]),
    ]
    result = compact(client, messages, "model")
    assert result == messages

def test_compact_long():
    client = MagicMock()
    # Create 10 messages
    messages = [types.Content(role="user", parts=[types.Part(text=str(i))]) for i in range(10)]
    
    # Mock the summarization
    client.models.generate_content.return_value.text = "Summary text"
    
    result = compact(client, messages, "model")
    
    # Should have 2 (summary + ack) + 4 (kept) = 6 messages
    assert len(result) == 2 + KEEP_LAST
    assert result[0].role == "user"
    assert "[Summary of earlier conversation" in result[0].parts[0].text
    assert "Summary text" in result[0].parts[0].text
    assert result[1].role == "model"
