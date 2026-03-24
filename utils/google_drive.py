import os

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    DRIVE_AVAILABLE = True
except ImportError:
    DRIVE_AVAILABLE = False


SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_drive_service():
    if not DRIVE_AVAILABLE:
        return None

    try:
        import streamlit as st

        # 優先使用 OAuth（一般 Gmail 帳號）
        if "oauth_refresh_token" in st.secrets:
            creds = Credentials(
                token=None,
                refresh_token=st.secrets["oauth_refresh_token"],
                token_uri="https://oauth2.googleapis.com/token",
                client_id=st.secrets["oauth_client_id"],
                client_secret=st.secrets["oauth_client_secret"],
                scopes=SCOPES,
            )
            creds.refresh(Request())
            return build("drive", "v3", credentials=creds)

        # 備用：Service Account
        from google.oauth2 import service_account
        creds_info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=SCOPES
        )
        return build("drive", "v3", credentials=creds)

    except Exception:
        return None


def upload_to_drive(file_path: str, file_name: str, folder_id: str | None = None):
    if not DRIVE_AVAILABLE:
        return None

    service = get_drive_service()
    if not service:
        return None

    folder_id = folder_id or os.getenv("DRIVE_FOLDER_ID")
    if not folder_id or folder_id == "PUT_YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE":
        return None

    mimetype = "application/octet-stream"
    lower_name = file_name.lower()
    if lower_name.endswith(".docx"):
        mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif lower_name.endswith(".pdf"):
        mimetype = "application/pdf"
    elif lower_name.endswith(".md"):
        mimetype = "text/markdown"
    elif lower_name.endswith(".txt"):
        mimetype = "text/plain"
    elif lower_name.endswith(".html"):
        mimetype = "text/html"

    file_metadata = {
        "name": file_name,
        "parents": [folder_id]
    }

    media = MediaFileUpload(file_path, mimetype=mimetype)
    created = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, webViewLink",
        supportsAllDrives=True,
    ).execute()

    return created
