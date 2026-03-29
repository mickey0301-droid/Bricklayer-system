"""
familiarity_manager.py
每個單字的熟悉度標記（熟悉 / 陌生 / 未標注）。
本地儲存於 data/familiarity_{language}.json，同步 GitHub 備份。
"""
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))

# 熟悉度常數
FAMILIAR   = "familiar"
UNFAMILIAR = "unfamiliar"

# 抽題權重：陌生 3× ，未標注 2× ，熟悉 1×
WEIGHTS = {
    FAMILIAR:   1.0,
    None:       2.0,
    UNFAMILIAR: 3.0,
}


def _data_path(language: str) -> str:
    os.makedirs(_DATA_DIR, exist_ok=True)
    return os.path.join(_DATA_DIR, f"familiarity_{language}.json")


def _gh_path(language: str) -> str:
    return f"data/familiarity_{language}.json"


# ── 載入 ──────────────────────────────────────────────────
def load_familiarity(language: str) -> dict:
    """回傳 {code_str: 'familiar'|'unfamiliar'} 的 dict。"""
    path = _data_path(language)
    # 1. 本地優先
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    # 2. GitHub fallback
    try:
        from utils.vocab_manager import _github_read
        content, _ = _github_read(_gh_path(language))
        if content:
            data = json.loads(content)
            if isinstance(data, dict):
                _write_local(path, data)
                return data
    except Exception:
        pass
    return {}


def _write_local(path: str, data: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ── 儲存 ──────────────────────────────────────────────────
def save_familiarity(language: str, data: dict):
    path = _data_path(language)
    _write_local(path, data)
    # 同步 GitHub（失敗靜默）
    try:
        from utils.vocab_manager import _github_write
        content_str = json.dumps(data, ensure_ascii=False, indent=2)
        _github_write(_gh_path(language), content_str, None,
                      f"Update familiarity: {language}")
    except Exception:
        pass


# ── 單字操作 ──────────────────────────────────────────────
def get_familiarity(language: str, code) -> str | None:
    """回傳 'familiar'、'unfamiliar' 或 None（未標注）。"""
    data = load_familiarity(language)
    return data.get(str(code))


def set_familiarity(language: str, code, status: str | None):
    """設定或清除一個單字的熟悉度。status 傳 None 表示清除標記。"""
    data = load_familiarity(language)
    key = str(code)
    if status is None:
        data.pop(key, None)
    else:
        data[key] = status
    save_familiarity(language, data)


# ── 抽題加權 ──────────────────────────────────────────────
def get_sample_weights(language: str, code_list: list) -> list:
    """
    給定 code 清單，回傳對應的抽題權重 list（配合 df.sample(weights=...)）。
    陌生 3× ，未標注 2× ，熟悉 1× 。
    """
    data = load_familiarity(language)
    result = []
    for code in code_list:
        status = data.get(str(code))
        result.append(WEIGHTS.get(status, WEIGHTS[None]))
    return result
