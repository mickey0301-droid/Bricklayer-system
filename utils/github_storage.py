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


def commit_file(local_path: Path, repo_path: str, message: str) -> bool:
    """
    Commit a local file to GitHub.

    Returns True on success, False if credentials are not configured or
    the API call fails.  Never raises — callers should treat failure as
    a silent no-op (data is still saved locally).
    """
    try:
        token, owner, repo, branch = _get_config()
        if not (token and owner and repo):
            return False  # No credentials configured; skip silently

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
        return r.status_code in (200, 201)

    except Exception:
        return False
