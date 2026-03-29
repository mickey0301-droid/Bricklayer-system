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


_BTN_STYLE = (
    "display:none;cursor:pointer;border:none;border-radius:50%;"
    "width:52px;height:52px;font-size:1.5rem;line-height:52px;text-align:center;"
    "background:#4A90D9;color:#fff;box-shadow:0 2px 6px rgba(0,0,0,.25);"
    "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"
)


def audio_player_dual(term_bytes: bytes, sentence_bytes: bytes) -> str:
    """Play term audio first, then sentence audio after term finishes.
    Falls back to a visible ▶ button when the browser blocks autoplay (mobile).
    """
    b64_term = base64.b64encode(term_bytes).decode()
    b64_sent = base64.b64encode(sentence_bytes).decode()
    uid = int(time.time() * 1000)
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;overflow:hidden;position:relative;height:60px">
<audio id="t{uid}">
  <source src="data:audio/mp3;base64,{b64_term}" type="audio/mp3">
</audio>
<audio id="s{uid}">
  <source src="data:audio/mp3;base64,{b64_sent}" type="audio/mp3">
</audio>
<button id="btn{uid}" style="{_BTN_STYLE}" title="播放">▶</button>
<script>
(function(){{
  var t = document.getElementById('t{uid}');
  var s = document.getElementById('s{uid}');
  var btn = document.getElementById('btn{uid}');
  function playSequence() {{
    t.currentTime = 0;
    s.currentTime = 0;
    t.onended = function() {{ setTimeout(function(){{ s.play(); }}, 400); }};
    t.play();
  }}
  btn.onclick = function() {{ btn.style.display='none'; playSequence(); }};
  var p = t.play();
  if (p !== undefined) {{
    p.then(function() {{
      t.onended = function() {{ setTimeout(function(){{ s.play(); }}, 400); }};
    }}).catch(function() {{
      btn.style.display = 'block';
    }});
  }}
}})();
</script>
</body></html>"""


def audio_player(audio_bytes: bytes) -> str:
    """Return a self-contained HTML document for use with components.html().
    Attempts autoplay; shows a ▶ button if the browser blocks it (mobile).
    """
    b64 = base64.b64encode(audio_bytes).decode()
    uid = int(time.time() * 1000)
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;overflow:hidden;position:relative;height:60px">
<audio id="a{uid}">
  <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
</audio>
<button id="btn{uid}" style="{_BTN_STYLE}" title="播放">▶</button>
<script>
(function(){{
  var a = document.getElementById('a{uid}');
  var btn = document.getElementById('btn{uid}');
  btn.onclick = function() {{ btn.style.display='none'; a.currentTime=0; a.play(); }};
  var p = a.play();
  if (p !== undefined) {{
    p.catch(function() {{ btn.style.display = 'block'; }});
  }}
}})();
</script>
</body></html>"""
