"""
GitHub-backed persistence for config files.
Commits config JSON files to the GitHub repo so data survives
Streamlit Cloud restarts (ephemeral filesystem).

Required keys in st.secrets (or environment variables):
  GITHUB_TOKEN  – Personal Access Token with repo Contents write access
  GITHUB_OWNER  – Repository owner login (e.g. "mickey0301-droid")
  GITHUB_REPO   – Repository name (e.g. "Briefings-System")
  GITHUB_BRANCH – Branch to commit to (default: "main")
"""
import base64
from pathlib import Path

import requests


def _get_config():
    token = owner = repo = branch = ""
    try:
        import streamlit as st
        token  = st.secrets.get("GITHUB_TOKEN",  "")
        owner  = st.secrets.get("GITHUB_OWNER",  "")
        repo   = st.secrets.get("GITHUB_REPO",   "")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
    except Exception:
        pass
    import os
    token  = token  or os.getenv("GITHUB_TOKEN",  "")
    owner  = owner  or os.getenv("GITHUB_OWNER",  "")
    repo   = repo   or os.getenv("GITHUB_REPO",   "")
    branch = branch or os.getenv("GITHUB_BRANCH", "main")
    return token, owner, repo, branch


def fetch_file(local_path: Path, repo_path: str) -> bool:
    """
    Download a file from GitHub and write it to local_path.

    Used on startup to restore ephemeral-filesystem files that were saved to
    the repo by a previous Streamlit session (via commit_file).

    Returns True if the file was successfully fetched and written.
    Returns False if credentials are missing, the file doesn't exist in the
    repo, or any network error occurs.  Never raises.
    """
    try:
        token, owner, repo, branch = _get_config()
        if not (token and owner and repo):
            return False

        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_path}"
        headers = {
            "Authorization": f"token {token}",
            "Accept":        "application/vnd.github.v3+json",
        }
        r = requests.get(api_url, headers=headers,
                         params={"ref": branch}, timeout=10)
        if r.status_code != 200:
            return False

        raw = r.json().get("content", "")
        content = base64.b64decode(raw).decode("utf-8")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")
        return True

    except Exception:
        return False


def commit_file(local_path: Path, repo_path: str, message: str) -> bool:
    """
    Commit a local file to GitHub.

    Returns True on success, False if credentials are not configured or
    the API call fails.  Never raises — callers should treat failure as
    a silent no-op (data is still saved locally).

    On failure the reason is stored in st.session_state["_github_sync_error"]
    so it can be displayed after st.rerun().
    """
    _set_sync_error = _make_sync_error_setter()
    try:
        token, owner, repo, branch = _get_config()
        if not (token and owner and repo):
            _set_sync_error("GitHub 憑證未設定（GITHUB_TOKEN / GITHUB_OWNER / GITHUB_REPO）")
            return False

        content  = local_path.read_text(encoding="utf-8")
        encoded  = base64.b64encode(content.encode("utf-8")).decode("ascii")
        api_url  = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_path}"
        headers  = {
            "Authorization": f"token {token}",
            "Accept":        "application/vnd.github.v3+json",
        }

        # Fetch current SHA (required to update an existing file)
        sha = None
        r = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=10)
        if r.status_code == 200:
            sha = r.json().get("sha")

        payload = {"message": message, "content": encoded, "branch": branch}
        if sha:
            payload["sha"] = sha

        r = requests.put(api_url, headers=headers, json=payload, timeout=15)
        ok = r.status_code in (200, 201)
        if not ok:
            _set_sync_error(f"GitHub API 回應 {r.status_code}：{r.text[:200]}")
        return ok

    except Exception as exc:
        _set_sync_error(f"例外錯誤：{exc}")
        return False


def _make_sync_error_setter():
    """Return a callable that stores a sync-error message in st.session_state if available."""
    def _set(msg: str):
        try:
            import streamlit as st
            st.session_state["_github_sync_error"] = msg
        except Exception:
            pass
    return _set
