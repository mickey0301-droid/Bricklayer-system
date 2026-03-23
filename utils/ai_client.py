import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 明確指定載入專案根目錄的 .env
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        f"找不到 OPENAI_API_KEY。請確認 {ENV_PATH} 存在，且裡面有正確的 OPENAI_API_KEY。"
    )

client = OpenAI(api_key=api_key)


def generate_briefing(prompt: str, model: str = "gpt-4.1-mini") -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2
    )
    return response.output_text