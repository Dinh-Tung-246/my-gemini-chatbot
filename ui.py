# ui.py
import streamlit as st
from config import AVAILABLE_MODELS, DEFAULT_MODEL # Import constants nếu có

def render_sidebar():
    """Xây dựng và hiển thị các thành phần trong sidebar."""
    st.sidebar.header("⚙️ Cấu hình")

    # Khởi tạo giá trị mặc định trong session state nếu chưa có
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'max_tokens' not in st.session_state:
        # Đặt giá trị mặc định ban đầu dựa trên model mặc định
        default_max = 8192 if "pro" in DEFAULT_MODEL else 2048
        st.session_state.max_tokens = default_max

    # Selectbox cho Model
    selected_model = st.sidebar.selectbox(
        "Chọn model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_model),
        key="model_selector_widget" # Dùng key khác để tránh xung đột nếu cần truy cập trực tiếp widget
    )
    # Cập nhật session state nếu model thay đổi
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Reset max_tokens khi model thay đổi để có giá trị mặc định phù hợp
        st.session_state.max_tokens = 8192 if "pro" in selected_model else 2048


    # Slider cho Temperature
    st.session_state.temperature = st.sidebar.slider(
        "🌡️ Temperature", 0.0, 1.0, st.session_state.temperature, 0.05, key="temp_slider"
    )

    # Slider cho Max Tokens (giá trị mặc định được lấy từ session state)
    st.session_state.max_tokens = st.sidebar.slider(
        "📊 Số token tối đa", 100, 8192, st.session_state.max_tokens, key="max_token_slider"
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Xoá lịch sử chat"):
        # Đặt lại các state liên quan đến chat
        st.session_state.messages = []
        st.session_state.chat = None
        st.session_state.current_model_name = None # Đảm bảo model và chat được load lại
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("Thay đổi model hoặc xóa lịch sử sẽ bắt đầu một cuộc trò chuyện mới.")


def display_messages():
    """Hiển thị lịch sử tin nhắn trong khu vực chat chính."""
    st.divider()
    if "messages" not in st.session_state or not st.session_state.messages:
        st.info("Chưa có tin nhắn nào. Bắt đầu trò chuyện thôi!")
        return # Không có gì để hiển thị

    for message in st.session_state.messages:
        if message.get("content"):
            role_display = "user" if message["role"] == "user" else "assistant"
            avatar_display = "👤" if role_display == "user" else "♊"
            with st.chat_message(role_display, avatar=avatar_display):
                st.markdown(message["content"])