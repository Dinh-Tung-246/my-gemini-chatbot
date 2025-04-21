import streamlit as st
import os
import google.generativeai as genai

# --- Cấu hình trang ---
st.set_page_config(page_title="Chatbot Gemini", page_icon="♊")
st.title("♊ Flash Chatbot V1.0.0")
st.caption("Dùng Google Gemini API để trả lời trực tiếp")

# --- Lấy API Key ---
google_api_key = os.environ.get("GOOGLE_API_KEY") or (st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') else None)
if not google_api_key:
    st.error("⚠️ Chưa có API Key. Hãy thêm vào biến môi trường hoặc Streamlit secrets.")
    st.stop()

# --- Cấu hình thư viện Google ---
try:
    genai.configure(api_key=google_api_key)
except Exception as e:
    st.error(f"Lỗi cấu hình Gemini: {e}")
    st.stop()

# --- Load model Gemini ---
@st.cache_resource
def load_model():
    model_name = "models/gemini-1.5-flash-latest"
    model = genai.GenerativeModel(model_name)
    return model

gemini_model = load_model()

# --- Giao diện Chat ---
st.divider()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn muốn hỏi gì về AI, công nghệ hay kiến thức tổng quát?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        msg_placeholder.markdown("🤖 Đang suy nghĩ...")

        try:
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]]

            response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
            if hasattr(response, 'text'):
                answer = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                answer = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            else:
                answer = "❌ Không thể tạo câu trả lời."

            msg_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"⚠️ Lỗi gọi API: {e}"
            msg_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
