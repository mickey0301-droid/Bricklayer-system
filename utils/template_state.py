from typing import Dict, Optional

import streamlit as st

from utils.report_templates import (
    get_default_template_id,
    get_safe_template,
    get_template_summary,
    resolve_template_id_from_name,
    resolve_template_name_from_id,
)


TEMPLATE_STATE_KEY = "selected_report_template_id"


def init_template_state() -> str:
    """
    初始化模板 session_state。
    若尚未設定，使用預設模板 ID。
    回傳目前有效的 template_id。
    """
    if TEMPLATE_STATE_KEY not in st.session_state:
        st.session_state[TEMPLATE_STATE_KEY] = get_default_template_id()

    current_id = st.session_state[TEMPLATE_STATE_KEY]
    safe_template = get_safe_template(template_id=current_id)
    safe_id = safe_template["id"]

    # 若原本 state 中是無效值，強制修正成安全值
    st.session_state[TEMPLATE_STATE_KEY] = safe_id
    return safe_id


def get_current_template_id() -> str:
    """
    取得目前 session_state 中的模板 ID。
    若尚未初始化，會自動初始化。
    """
    return init_template_state()


def get_current_template_name() -> str:
    """
    取得目前模板的 UI 顯示名稱。
    """
    current_id = init_template_state()
    return resolve_template_name_from_id(current_id)


def set_current_template_by_id(template_id: Optional[str]) -> str:
    """
    以 template_id 更新目前模板。
    若傳入值無效，會自動回退到預設模板。
    回傳最終有效的 template_id。
    """
    safe_template = get_safe_template(template_id=template_id)
    safe_id = safe_template["id"]
    st.session_state[TEMPLATE_STATE_KEY] = safe_id
    return safe_id


def set_current_template_by_name(template_name: Optional[str]) -> str:
    """
    以 UI 顯示名稱更新目前模板。
    若傳入值無效，會自動回退到預設模板。
    回傳最終有效的 template_id。
    """
    resolved_id = resolve_template_id_from_name(template_name)
    return set_current_template_by_id(resolved_id)


def get_current_template() -> Dict:
    """
    取得目前完整模板資料。
    """
    current_id = init_template_state()
    return get_safe_template(template_id=current_id)


def get_current_template_summary() -> Dict:
    """
    取得目前模板摘要資料，方便 UI 顯示或 debug。
    """
    current_id = init_template_state()
    return get_template_summary(template_id=current_id)