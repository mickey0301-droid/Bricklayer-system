import json
from pathlib import Path
from typing import Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_CONFIG_PATH = BASE_DIR / "config" / "report_templates.json"


def load_report_templates() -> Dict:
    """
    讀取報告模板設定檔。
    若檔案不存在或格式錯誤，直接拋出例外，方便開發時快速發現問題。
    """
    if not TEMPLATE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"找不到模板設定檔：{TEMPLATE_CONFIG_PATH}")

    with open(TEMPLATE_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "templates" not in data or not isinstance(data["templates"], list):
        raise ValueError("report_templates.json 格式錯誤：缺少 templates 陣列")

    if "default_template" not in data:
        raise ValueError("report_templates.json 格式錯誤：缺少 default_template")

    return data


def get_all_templates() -> List[Dict]:
    """
    回傳所有模板清單。
    """
    data = load_report_templates()
    return data.get("templates", [])


def get_default_template_id() -> str:
    """
    回傳預設模板 ID。
    """
    data = load_report_templates()
    return data["default_template"]


def get_template_by_id(template_id: str) -> Optional[Dict]:
    """
    依 template_id 取得單一模板。
    找不到時回傳 None。
    """
    templates = get_all_templates()
    for template in templates:
        if template.get("id") == template_id:
            return template
    return None


def get_default_template() -> Dict:
    """
    回傳預設模板完整資料。
    若預設模板不存在，拋出例外。
    """
    default_id = get_default_template_id()
    template = get_template_by_id(default_id)

    if template is None:
        raise ValueError(f"找不到預設模板：{default_id}")

    return template


def get_template_options() -> List[str]:
    """
    回傳給 UI selectbox 使用的模板名稱清單。
    例如：
    ["Strategic Briefing", "Policy Memo", "Academic Analysis"]
    """
    templates = get_all_templates()
    return [template["name"] for template in templates]


def get_template_name_to_id_map() -> Dict[str, str]:
    """
    回傳 UI 顯示名稱 -> template_id 的對照表。
    """
    templates = get_all_templates()
    return {template["name"]: template["id"] for template in templates}


def get_template_id_to_name_map() -> Dict[str, str]:
    """
    回傳 template_id -> UI 顯示名稱 的對照表。
    """
    templates = get_all_templates()
    return {template["id"]: template["name"] for template in templates}


def get_default_template_name() -> str:
    """
    回傳預設模板的 UI 顯示名稱。
    例如：Strategic Briefing
    """
    default_template = get_default_template()
    return default_template["name"]


def resolve_template_id_from_name(template_name: Optional[str]) -> str:
    """
    將 UI 顯示名稱轉成 template_id。
    若名稱不存在、為空值，則回退到預設模板 ID。
    """
    if not template_name:
        return get_default_template_id()

    name_to_id = get_template_name_to_id_map()
    return name_to_id.get(template_name, get_default_template_id())


def resolve_template_name_from_id(template_id: Optional[str]) -> str:
    """
    將 template_id 轉成 UI 顯示名稱。
    若 id 不存在、為空值，則回退到預設模板名稱。
    """
    if not template_id:
        return get_default_template_name()

    id_to_name = get_template_id_to_name_map()
    return id_to_name.get(template_id, get_default_template_name())


def get_safe_template(template_id: Optional[str] = None, template_name: Optional[str] = None) -> Dict:
    """
    安全取得模板。
    優先使用 template_id，其次使用 template_name。
    若兩者都無效，則回退到預設模板。
    """
    if template_id:
        template = get_template_by_id(template_id)
        if template is not None:
            return template

    if template_name:
        resolved_id = resolve_template_id_from_name(template_name)
        template = get_template_by_id(resolved_id)
        if template is not None:
            return template

    return get_default_template()


def build_template_instruction(template_id: Optional[str] = None) -> str:
    """
    將模板設定轉成一段可提供給 AI 的指示文字。
    若未提供 template_id，則使用預設模板。
    """
    template = get_safe_template(template_id=template_id)

    template_name = template.get("name", "")
    description = template.get("description", "")
    tone = template.get("tone", "")
    style_prompt = template.get("style_prompt", "")
    sections = template.get("sections", [])

    section_lines = "\n".join([f"- {section}" for section in sections])

    instruction = f"""【報告模板設定】
模板名稱：{template_name}
模板說明：{description}
寫作語氣：{tone}

【模板寫作要求】
{style_prompt}

【建議章節結構】
{section_lines}
"""

    return instruction.strip()


def get_template_summary(template_id: Optional[str] = None) -> Dict:
    """
    回傳模板摘要資訊，方便 UI 或日後 debug 使用。
    """
    template = get_safe_template(template_id=template_id)

    return {
        "id": template.get("id", ""),
        "name": template.get("name", ""),
        "description": template.get("description", ""),
        "tone": template.get("tone", ""),
        "sections": template.get("sections", []),
    }