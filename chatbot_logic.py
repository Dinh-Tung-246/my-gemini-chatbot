# chatbot_logic.py
import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm

@st.cache_resource
def load_gemini_model(model_name):
    """Tải và cache model Generative AI."""
    try:
        # st.info(f"Đang tải model: {model_name}...")
        model = genai.GenerativeModel(model_name)
        # st.success(f"Đã tải thành công model: {model_name}")
        # Cập nhật model đang được cache trong session_state
        st.session_state.current_model_name = model_name
        return model
    except Exception as e:
        st.error(f"❌ Lỗi nghiêm trọng khi tải model {model_name}: {e}")
        st.stop()

def initialize_chat_session(model):
    """
    Khởi tạo hoặc tải lại ChatSession nếu cần (ví dụ: model thay đổi).
    Cập nhật st.session_state.chat và st.session_state.messages nếu cần.
    """
    model_name_selected = st.session_state.get('selected_model', None)
    current_model_loaded = st.session_state.get('current_model_name', None)
    chat_session_exists = "chat" in st.session_state and st.session_state.chat is not None

    # Kiểm tra các điều kiện cần khởi tạo lại chat
    model_changed = current_model_loaded != model_name_selected
    needs_reinitialization = not chat_session_exists or model_changed

    if needs_reinitialization:
        if model_changed and chat_session_exists:
            st.warning(f"Model đã thay đổi. Bắt đầu cuộc trò chuyện mới với {model_name_selected}.")
            st.session_state.messages = [] # Xóa lịch sử khi đổi model
            st.session_state.chat = None # Xóa session cũ

        # Khởi tạo tin nhắn chào mừng nếu messages rỗng hoặc vừa bị xóa
        if not st.session_state.get("messages"):
             model_short_name = model_name_selected.split('/')[-1]
             st.session_state.messages = [{
                 "role": "model",
                 "content": f"🤖 Xin chào! Mình là **{model_short_name}**. Bạn muốn hỏi gì?"
             }]

        # Chuyển đổi lịch sử hiện có (trừ tin nhắn cuối cùng) sang định dạng Google AI
        google_history = []
        if st.session_state.get("messages"):
            for msg in st.session_state.messages[:-1]:
                if msg.get("content"):
                    role_google = "user" if msg["role"] == "user" else "model"
                    try:
                         # Sử dụng class Content và Part từ glm
                         google_history.append(glm.Content(role=role_google, parts=[glm.Part(text=msg["content"])]))
                    except Exception as e:
                        st.warning(f"Bỏ qua tin nhắn không hợp lệ khi tạo history: {msg}. Lỗi: {e}")

        # Khởi tạo chat session mới
        try:
            st.session_state.chat = model.start_chat(history=google_history)
            # Cập nhật lại model đang dùng của session này
            st.session_state.current_model_name = model_name_selected
            # st.info("Chat session đã được khởi tạo/tải lại.")
        except Exception as e:
            st.error(f"❌ Lỗi nghiêm trọng khi bắt đầu chat session: {e}")
            st.stop()
    # Trả về chat session hiện tại (dù mới tạo hay đã có)
    return st.session_state.chat


def generate_response_stream(chat_session, prompt, generation_config, safety_settings):
    """
    Gửi prompt đến chat session và trả về một generator cho response stream.
    Xử lý các lỗi API cụ thể trong quá trình gửi.
    """
    try:
        response_stream = chat_session.send_message(
            prompt,
            stream=True,
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        # Trả về generator để xử lý bên ngoài
        for chunk in response_stream:
            yield chunk

    except genai.types.BlockedPromptException as e:
        st.error(f"🚫 Yêu cầu bị chặn: {e}")
        yield f"⚠️ **Lỗi:** Yêu cầu của bạn đã bị chặn bởi bộ lọc nội dung." # Trả về thông báo lỗi như một chunk
    except genai.types.StopCandidateException as e:
        st.error(f"⚠️ Phản hồi bị dừng: {e}")
        yield f"⚠️ **Lỗi:** Phản hồi bị dừng đột ngột bởi API."
    except Exception as e:
        st.error(f"❌ Lỗi khi gọi API: {e}")
        yield f"⚠️ **Lỗi:** Đã có lỗi xảy ra khi giao tiếp với Gemini API."

def get_blocked_reason(chat_session):
    """Cố gắng lấy lý do bị chặn từ phản hồi cuối cùng của chat session."""
    try:
        last_response = chat_session.last_response
        if last_response and last_response.prompt_feedback:
            return last_response.prompt_feedback.block_reason.name
    except Exception:
        pass # Bỏ qua nếu không lấy được
    return "Không rõ lý do"