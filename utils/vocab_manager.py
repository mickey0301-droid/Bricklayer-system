import os
import json
import base64
import pandas as pd

DATA_FOLDER = "data"
VOCAB_COLUMNS = ["code", "term", "reading", "meaning", "pos", "note"]


def ensure_data_folder():
    os.makedirs(DATA_FOLDER, exist_ok=True)


def get_vocab_path(language: str) -> str:
    return os.path.join(DATA_FOLDER, f"{language}_vocab.json")


def get_pattern_vocab_path(language: str) -> str:
    return os.path.join(DATA_FOLDER, f"{language}_pattern_vocab.json")


def empty_vocab_df() -> pd.DataFrame:
    return pd.DataFrame(columns=VOCAB_COLUMNS)


def normalize_vocab_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_vocab_df()
    df = df.copy()
    for col in VOCAB_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[VOCAB_COLUMNS]
    df = df.fillna("")
    for col in VOCAB_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def ensure_min_rows(df: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    df = normalize_vocab_df(df)
    if len(df) >= min_rows:
        return df
    rows_to_add = min_rows - len(df)
    empty_rows = pd.DataFrame(
        [[""] * len(VOCAB_COLUMNS) for _ in range(rows_to_add)],
        columns=VOCAB_COLUMNS
    )
    return pd.concat([df, empty_rows], ignore_index=True)


# ── GitHub API helpers ────────────────────────────────────

def _github_config():
    """從 Streamlit secrets 取得 GitHub token、repo 和 branch。"""
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN", "")
        # 支援 Briefings 格式（GITHUB_OWNER + GITHUB_REPO 分開）
        owner = st.secrets.get("GITHUB_OWNER", "")
        repo_name = st.secrets.get("GITHUB_REPO", "")
        if owner and repo_name and "/" not in repo_name:
            repo = f"{owner}/{repo_name}"
        else:
            # 相容舊格式（owner/repo 合在一起）
            repo = repo_name or "mickey0301-droid/Bricklayer-System"
        return token, repo
    except Exception:
        return "", ""


def _github_branch():
    """從 Streamlit secrets 取得 branch，預設 main。"""
    try:
        import streamlit as st
        return st.secrets.get("GITHUB_BRANCH", "main")
    except Exception:
        return "main"


def _gh_headers() -> dict:
    token, _ = _github_config()
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _github_read(gh_path: str):
    """
    讀取 GitHub 上的 JSON 檔案，回傳 (data, sha) 或 (None, None)。
    先嘗試 Contents API（≤1MB），超過大小時改用 Git Data API（支援到 100MB）。
    """
    import requests
    token, repo = _github_config()
    if not token:
        return None, None
    branch = _github_branch()
    headers = _gh_headers()

    try:
        # 嘗試 Contents API
        url = f"https://api.github.com/repos/{repo}/contents/{gh_path}?ref={branch}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            d = r.json()
            encoding = d.get("encoding", "base64")
            sha = d.get("sha")
            if encoding == "base64" and d.get("content"):
                content = base64.b64decode(d["content"]).decode("utf-8")
                return json.loads(content), sha
            # encoding == "none" → 檔案太大，改用 Git Blob API
            if sha:
                blob_url = f"https://api.github.com/repos/{repo}/git/blobs/{sha}"
                br = requests.get(blob_url, headers=headers, timeout=30)
                if br.status_code == 200:
                    bd = br.json()
                    content = base64.b64decode(bd["content"]).decode("utf-8")
                    return json.loads(content), sha
        elif r.status_code == 403:
            # 檔案 > 1MB，用 Git Trees API 找 blob SHA，再用 Blob API 讀取
            trees_url = (
                f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
            )
            tr = requests.get(trees_url, headers=headers, timeout=15)
            if tr.status_code == 200:
                for item in tr.json().get("tree", []):
                    if item["path"] == gh_path:
                        blob_sha = item["sha"]
                        blob_url = f"https://api.github.com/repos/{repo}/git/blobs/{blob_sha}"
                        br = requests.get(blob_url, headers=headers, timeout=30)
                        if br.status_code == 200:
                            bd = br.json()
                            content = base64.b64decode(bd["content"]).decode("utf-8")
                            return json.loads(content), blob_sha
    except Exception:
        pass
    return None, None


def _github_write(gh_path: str, content_str: str, _sha_unused, message: str) -> tuple:
    """
    用 Git Data API 寫入 GitHub，支援任意大小。
    不依賴檔案本身的 SHA，完全繞開 Contents API 的 1MB 限制。
    回傳 (success: bool, error_msg: str)。
    """
    import requests
    token, repo = _github_config()
    if not token:
        return False, "未設定 GITHUB_TOKEN"
    branch = _github_branch()
    headers = _gh_headers()

    try:
        # Step 1：建立 blob
        blob_r = requests.post(
            f"https://api.github.com/repos/{repo}/git/blobs",
            headers=headers,
            json={
                "content": base64.b64encode(content_str.encode("utf-8")).decode("utf-8"),
                "encoding": "base64",
            },
            timeout=30,
        )
        if blob_r.status_code != 201:
            return False, f"建立 blob 失敗 HTTP {blob_r.status_code}: {blob_r.text[:200]}"
        blob_sha = blob_r.json()["sha"]

        # Step 2：取得目前 branch 的 commit SHA
        ref_r = requests.get(
            f"https://api.github.com/repos/{repo}/git/ref/heads/{branch}",
            headers=headers, timeout=10,
        )
        if ref_r.status_code != 200:
            return False, f"取得 ref 失敗 HTTP {ref_r.status_code}: {ref_r.text[:200]}"
        latest_commit_sha = ref_r.json()["object"]["sha"]

        # Step 3：取得目前 commit 的 tree SHA
        commit_r = requests.get(
            f"https://api.github.com/repos/{repo}/git/commits/{latest_commit_sha}",
            headers=headers, timeout=10,
        )
        if commit_r.status_code != 200:
            return False, f"取得 commit 失敗 HTTP {commit_r.status_code}"
        base_tree_sha = commit_r.json()["tree"]["sha"]

        # Step 4：建立新 tree（只更新目標檔案）
        tree_r = requests.post(
            f"https://api.github.com/repos/{repo}/git/trees",
            headers=headers,
            json={
                "base_tree": base_tree_sha,
                "tree": [{"path": gh_path, "mode": "100644", "type": "blob", "sha": blob_sha}],
            },
            timeout=30,
        )
        if tree_r.status_code != 201:
            return False, f"建立 tree 失敗 HTTP {tree_r.status_code}: {tree_r.text[:200]}"
        new_tree_sha = tree_r.json()["sha"]

        # Step 5：建立新 commit
        new_commit_r = requests.post(
            f"https://api.github.com/repos/{repo}/git/commits",
            headers=headers,
            json={"message": message, "tree": new_tree_sha, "parents": [latest_commit_sha]},
            timeout=15,
        )
        if new_commit_r.status_code != 201:
            return False, f"建立 commit 失敗 HTTP {new_commit_r.status_code}: {new_commit_r.text[:200]}"
        new_commit_sha = new_commit_r.json()["sha"]

        # Step 6：更新 branch ref
        update_r = requests.patch(
            f"https://api.github.com/repos/{repo}/git/refs/heads/{branch}",
            headers=headers,
            json={"sha": new_commit_sha},
            timeout=15,
        )
        if update_r.status_code != 200:
            return False, f"更新 ref 失敗 HTTP {update_r.status_code}: {update_r.text[:200]}"

        return True, ""

    except Exception as e:
        return False, str(e)


# ── 詞庫讀寫 ─────────────────────────────────────────────

def load_vocab(language: str) -> pd.DataFrame:
    ensure_data_folder()
    path = get_vocab_path(language)
    gh_path = f"data/{language}_vocab.json"

    # 1. GitHub 優先：GitHub 是唯一可信來源，每次都從 GitHub 載入
    token, _ = _github_config()
    if token:
        gh_data, _ = _github_read(gh_path)
        if gh_data:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(gh_data, f, ensure_ascii=False, indent=2)
            return normalize_vocab_df(pd.DataFrame(gh_data))

    # 2. GitHub 無法連線時，退而使用本機快取（離線保底）
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                local_data = json.load(f)
            if local_data:
                return normalize_vocab_df(pd.DataFrame(local_data))
        except Exception:
            pass

    return empty_vocab_df()


def auto_assign_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    對沒有填寫 code（空白/非數字）的列，自動補上下一個可用的號碼。
    已有數字的列保留原本的 code，不重新編號（避免打亂例句快取）。
    """
    df = df.copy()

    # 找出目前最大的 code
    max_code = 0
    for v in df["code"]:
        try:
            n = int(str(v).strip())
            if n > max_code:
                max_code = n
        except (ValueError, TypeError):
            pass

    # 依序補上空白列的 code
    next_code = max_code + 1
    for i in df.index:
        v = str(df.at[i, "code"]).strip()
        valid = False
        try:
            int(v)
            valid = True
        except (ValueError, TypeError):
            pass
        if not valid:
            df.at[i, "code"] = str(next_code)
            next_code += 1

    return df


def save_vocab(language: str, df: pd.DataFrame):
    ensure_data_folder()
    path = get_vocab_path(language)
    gh_path = f"data/{language}_vocab.json"

    df = normalize_vocab_df(df)
    # 去掉完全空白的列
    mask = df.apply(lambda row: any(str(v).strip() != "" for v in row), axis=1)
    df = df[mask].copy()
    # 自動補全空白的 code
    df = auto_assign_codes(df)

    records = df.to_dict(orient="records")
    content_str = json.dumps(records, ensure_ascii=False, indent=2)

    # 1. 存到本地（快）
    with open(path, "w", encoding="utf-8") as f:
        f.write(content_str)

    # 2. 同步到 GitHub（讓資料跨部署持久化）
    token, _ = _github_config()
    if token:
        ok, err = _github_write(
            gh_path, content_str, None,
            f"Update {language} vocabulary ({len(records)} entries)"
        )
        if not ok:
            raise RuntimeError(f"GitHub 同步失敗：{err}")


# ── 句型詞庫讀寫（獨立於主詞庫）──────────────────────────

def load_pattern_vocab(language: str) -> pd.DataFrame:
    ensure_data_folder()
    path = get_pattern_vocab_path(language)
    gh_path = f"data/{language}_pattern_vocab.json"

    # 1. GitHub 優先：GitHub 是唯一可信來源，每次都從 GitHub 載入
    token, _ = _github_config()
    if token:
        gh_data, _ = _github_read(gh_path)
        if gh_data:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(gh_data, f, ensure_ascii=False, indent=2)
            return normalize_vocab_df(pd.DataFrame(gh_data))

    # 2. GitHub 無法連線時，退而使用本機快取（離線保底）
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                local_data = json.load(f)
            if local_data:
                return normalize_vocab_df(pd.DataFrame(local_data))
        except Exception:
            pass

    return empty_vocab_df()


def save_pattern_vocab(language: str, df: pd.DataFrame):
    ensure_data_folder()
    path = get_pattern_vocab_path(language)
    gh_path = f"data/{language}_pattern_vocab.json"

    df = normalize_vocab_df(df)
    mask = df.apply(lambda row: any(str(v).strip() != "" for v in row), axis=1)
    df = df[mask].copy()
    df = auto_assign_codes(df)

    records = df.to_dict(orient="records")
    content_str = json.dumps(records, ensure_ascii=False, indent=2)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content_str)

    token, _ = _github_config()
    if token:
        ok, err = _github_write(
            gh_path, content_str, None,
            f"Update {language} pattern vocabulary ({len(records)} entries)"
        )
        if not ok:
            raise RuntimeError(f"GitHub 同步失敗：{err}")
