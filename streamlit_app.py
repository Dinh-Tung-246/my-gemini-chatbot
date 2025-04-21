# streamlit_app.py
import streamlit as st
import google.generativeai as genai

# Import các module đã tách
from config import load_api_key, configure_gemini
from ui import render_sidebar, display_messages
from chatbot_logic import load_gemini_model, initialize_chat_session, generate_response_stream, get_blocked_reason

# --- 1. Cấu hình trang & API Key ---
st.set_page_config(page_title="Modular Chatbot V1.1", page_icon="🧩", layout="wide")
st.title("🧩 Modular Gemini Chatbot")
st.caption("Chatbot được tổ chức thành nhiều file")

api_key = load_api_key()
configure_gemini(api_key)

# --- 2. Render Sidebar & Lấy Cấu hình ---
render_sidebar() # Hàm này cập nhật session_state

# Lấy các giá trị cấu hình từ session_state (do sidebar cập nhật)
selected_model_name = st.session_state.selected_model
temperature = st.session_state.temperature
max_tokens = st.session_state.max_tokens

# --- 3. Load Model & Khởi tạo Chat ---
# Model được cache, chỉ load lại nếu selected_model_name thay đổi
model = load_gemini_model(selected_model_name)

# Khởi tạo chat session, hàm này sẽ kiểm tra và tạo mới nếu cần
chat = initialize_chat_session(model)

# --- 4. Hiển thị Lịch sử Chat ---
display_messages()

# --- 5. Xử lý Input & Gọi API ---
if prompt := st.chat_input("💬 Nhập câu hỏi của bạn..."):
    # Thêm tin nhắn người dùng vào state và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Chuẩn bị gọi API
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    safety_settings = [
        {"category": c, "threshold": "BLOCK_NONE"} for c in [
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
    ]

    # Hiển thị placeholder và gọi hàm tạo response stream
    with st.chat_message("assistant", avatar="♊"):
        message_placeholder = st.empty()
        message_placeholder.markdown("♊ Đang suy nghĩ...")
        full_response = ""
        assistant_response_error = False # Cờ để kiểm tra lỗi trong stream

        # Sử dụng generator từ chatbot_logic
        response_generator = generate_response_stream(
            chat, prompt, generation_config, safety_settings
        )

        for chunk in response_generator:
            if isinstance(chunk, str) and chunk.startswith("⚠️ **Lỗi:**"): # Kiểm tra nếu chunk là thông báo lỗi từ generator
                full_response = chunk # Ghi đè bằng thông báo lỗi
                assistant_response_error = True
                break # Dừng xử lý stream nếu có lỗi
            elif hasattr(chunk, 'text') and chunk.text:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
            # Có thể thêm xử lý cho các loại chunk khác nếu cần (ví dụ: function calls)

        # Xử lý sau khi stream kết thúc hoặc bị lỗi
        if not full_response and not assistant_response_error: # Stream xong nhưng không có text VÀ không phải lỗi từ generator
            block_reason = get_blocked_reason(chat)
            full_response = f"⚠️ Rất tiếc, không thể tạo phản hồi. Lý do có thể là: **{block_reason}**."
            assistant_response_error = True # Đánh dấu là có vấn đề

        # Hiển thị kết quả cuối cùng
        message_placeholder.markdown(full_response)

        # Chỉ thêm vào lịch sử nếu không có lỗi VÀ có nội dung
        if not assistant_response_error and full_response:
            st.session_state.messages.append({"role": "model", "content": full_response})
        elif assistant_response_error:
             # Nếu có lỗi, vẫn thêm tin nhắn lỗi vào lịch sử để người dùng biết
             # Kiểm tra xem tin nhắn cuối có phải user không
             if st.session_state.messages[-1]["role"] == "user":
                 st.session_state.messages.append({"role": "model", "content": full_response})
             else: # Nếu tin nhắn cuối là lỗi cũ, cập nhật nó
                 st.session_state.messages[-1] = {"role": "model", "content": full_response}