# config.py
import streamlit as st
import os
import google.generativeai as genai

def load_api_key():
    """
    Tải Google API Key từ biến môi trường hoặc Streamlit secrets.
    Ưu tiên biến môi trường. Dừng ứng dụng nếu không tìm thấy key.
    """
    google_api_key = None
    google_api_key_env = os.environ.get("GOOGLE_API_KEY")

    if google_api_key_env:
        google_api_key = google_api_key_env
    else:
        try:
            if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
                google_api_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            pass # Bỏ qua lỗi nếu không tìm thấy secrets

    if not google_api_key:
        st.error("⚠️ **GOOGLE_API_KEY không được tìm thấy!**")
        st.info(
            "**Để chạy local:** Đặt biến môi trường `GOOGLE_API_KEY`.\n"
            "**Để deploy:** Thêm `GOOGLE_API_KEY` vào Streamlit Secrets."
            # Thêm hướng dẫn chi tiết hơn nếu muốn
        )
        st.stop()
    return google_api_key

def configure_gemini(api_key):
    """Cấu hình thư viện Google Generative AI."""
    try:
        genai.configure(api_key=api_key)
        # st.success("Đã cấu hình Gemini thành công.") # Optional: for debugging
    except Exception as e:
        st.error(f"❌ Lỗi cấu hình thư viện Gemini: {e}")
        st.stop()

# Các hằng số cấu hình khác có thể đặt ở đây nếu cần
AVAILABLE_MODELS = ["models/gemini-1.5-flash-latest", "models/gemini-1.5-pro-latest"]
DEFAULT_MODEL = AVAILABLE_MODELS[0]