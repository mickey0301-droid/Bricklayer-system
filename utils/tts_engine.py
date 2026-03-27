import base64
import time
import streamlit as st
from openai import OpenAI


def _require_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("尚未設定 OPENAI_API_KEY。請在 .streamlit/secrets.toml 裡加入 API key。")
    return OpenAI(api_key=api_key)


# Language-specific TTS instructions to prevent the model from
# defaulting to the wrong language when the text contains CJK characters.
_TTS_INSTRUCTIONS = {
    "japanese": (
        "You are reading Japanese text aloud. "
        "Read every character as Japanese — do NOT read any kanji or kana as Chinese. "
        "Use natural Japanese pronunciation and intonation throughout."
    ),
    "korean": (
        "You are reading Korean text aloud. "
        "Read every character as Korean with natural Korean pronunciation and intonation."
    ),
    "spanish": (
        "You are reading Spanish text aloud with natural Spanish pronunciation and intonation."
    ),
    "french": (
        "You are reading French text aloud with natural French pronunciation and intonation."
    ),
    "german": (
        "You are reading German text aloud with natural German pronunciation and intonation."
    ),
}

_DEFAULT_INSTRUCTION = "Read the text aloud naturally in the appropriate language."


def generate_tts_audio(text: str, language: str) -> bytes:
    client = _require_openai_client()

    instructions = _TTS_INSTRUCTIONS.get(language, _DEFAULT_INSTRUCTION)

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        instructions=instructions,
    )

    return response.read()


def audio_player(audio_bytes: bytes) -> str:
    """Return a self-contained HTML document for use with components.html().
    The iframe reloads its srcdoc on every call, so the script runs fresh
    each time and .play() is called reliably on every button press.
    """
    b64 = base64.b64encode(audio_bytes).decode()
    uid = int(time.time() * 1000)
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;overflow:hidden">
<audio id="a{uid}" style="display:none">
  <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
</audio>
<script>
  var a = document.getElementById('a{uid}');
  if (a) a.play();
</script>
</body></html>"""
