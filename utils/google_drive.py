import os
import pickle

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
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

    creds = None

    try:
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists("credentials.json"):
                    return None
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)

            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)

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
        fields="id, webViewLink"
    ).execute()

    return created