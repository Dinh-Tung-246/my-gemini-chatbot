# streamlit_app.py (Hoàn chỉnh - Sửa lỗi SecretNotFound & Tùy chọn Key cố định)

import streamlit as st
import os
import google.generativeai as genai # Thư viện gốc Google
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.llms.custom import CustomLLM # Lớp cơ sở cho LLM tùy chỉnh
from llama_index.core.embeddings import BaseEmbedding # Lớp cơ sở cho Embedding tùy chỉnh
from llama_index.core.llms.llm import LLM
from typing import Any, List, Optional, Sequence, AsyncGenerator
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.schema import BaseNode
# from tqdm import tqdm # Không dùng trong streamlit cache

# --- Cấu hình trang Streamlit ---
st.set_page_config(page_title="Chatbot Gemini", page_icon="♊", layout="centered")
st.title("♊ Chatbot LlamaIndex & Google Gemini")
st.caption("Được xây dựng bằng LlamaIndex và Streamlit")

# --- Google API Key Configuration ---
# TÙY CHỌN 1: Đặt Key trực tiếp vào code (CHỈ DÙNG ĐỂ TEST LOCAL - KHÔNG AN TOÀN!)
# Bỏ comment dòng dưới và thay YOUR_GOOGLE_API_KEY_HERE bằng key thật của bạn.
# Nhớ comment hoặc xóa dòng này trước khi chia sẻ code hoặc deploy!
FIXED_API_KEY = "AIzaSyC4eCRIkm0MIUkBtMkZmXSe-BOrk2qySnY" # Key cố định của bạn

# TÙY CHỌN 2: Sử dụng biến môi trường hoặc Streamlit Secrets (AN TOÀN HƠN)
google_api_key = None
secrets_available = hasattr(st, 'secrets') # Kiểm tra xem st.secrets có tồn tại không

# ƯU TIÊN KEY CỐ ĐỊNH NẾU NÓ ĐƯỢC ĐỊNH NGHĨA VÀ KHÔNG PHẢI PLACEHOLDER
if 'FIXED_API_KEY' in locals() and FIXED_API_KEY and FIXED_API_KEY != "YOUR_GOOGLE_API_KEY_HERE":
    google_api_key = FIXED_API_KEY
    # Chỉ hiện cảnh báo một lần khi dùng key cố định
    if "fixed_key_warning_shown" not in st.session_state:
        # st.warning("Đang sử dụng API Key đặt cố định trong code (KHÔNG AN TOÀN!).", icon="⚠️")
        st.session_state.fixed_key_warning_shown = True # Đánh dấu đã hiển thị
else:
    # Nếu không dùng key cố định, thử lấy từ biến môi trường
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        # st.info("Đã lấy API Key từ biến môi trường.", icon="🖥️")
        pass # Đã lấy được từ env var, không cần làm gì thêm
    elif secrets_available:
        # Nếu không có env var, mới thử lấy từ secrets (dùng get với default)
        google_api_key = st.secrets.get("GOOGLE_API_KEY", None)
        if google_api_key:
            st.info("Đã lấy API Key từ Streamlit Secrets.", icon="🔒")

# Nếu sau tất cả các bước trên vẫn không có key, cho phép nhập
if not google_api_key:
    st.warning("Không tìm thấy GOOGLE_API_KEY.")
    google_api_key = st.text_input("Nhập Google API Key:", type="password", key="api_key_input")

# Kiểm tra lần cuối
if not google_api_key:
    st.error("Vui lòng cung cấp Google API Key.", icon="🚨")
    st.stop() # Dừng ứng dụng nếu không có key

# --- Cấu hình thư viện Google ---
try:
    genai.configure(api_key=google_api_key)
except Exception as e:
    st.error(f"Lỗi cấu hình thư viện Google: {e}", icon="🔥")
    st.stop()

# --- Cache Models và Khởi tạo ---
@st.cache_data
def load_available_models():
    llm_models, embed_models = [], []
    try:
        # st.info("Đang kiểm tra models khả dụng...") # Bỏ bớt thông báo
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods: llm_models.append(m.name)
            if 'embedContent' in m.supported_generation_methods: embed_models.append(m.name)
        # st.success("Kiểm tra model hoàn tất.")
    except Exception as e:
        st.warning(f"Không thể liệt kê models: {e}. Dùng mặc định.")
        llm_models, embed_models = ["models/gemini-1.5-flash-latest"], ["models/embedding-001"]
    # In ra console để debug (không hiển thị trên UI)
    print(f"Available LLM models: {llm_models}")
    print(f"Available Embedding models: {embed_models}")
    return llm_models, embed_models

available_llm_models, available_embedding_models = load_available_models()

# --- Định nghĩa Lớp LLM Tùy chỉnh ---
class MyGeminiLLM(CustomLLM):
    model_name: str = "models/gemini-1.5-flash-latest"
    _model: Any = None
    def __init__(self, model_name: str = "models/gemini-1.5-flash-latest", **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        try:
            if self.model_name not in available_llm_models:
                 print(f"Warning: Model '{self.model_name}' not in available LLMs {available_llm_models}. Trying fallback.")
                 if available_llm_models: self.model_name = available_llm_models[0]; print(f"Switched to: {self.model_name}")
                 else: raise ValueError("No fallback LLM available.")
            self._model = genai.GenerativeModel(self.model_name)
            print(f"MyGeminiLLM initialized with model: {self.model_name}")
        except Exception as e: print(f"Error initializing LLM '{self.model_name}': {e}"); raise
    @property
    def metadata(self) -> LLMMetadata:
        context_window = 1048576 if "1.5" in self.model_name else 30720
        return LLMMetadata(context_window=context_window, num_output=8192, model_name=self.model_name, is_chat_model=True)
    def _generate_sync(self, prompt: str, **kwargs) -> CompletionResponse:
         try:
             safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
             response = self._model.generate_content(prompt, safety_settings=safety_settings)
             generated_text = ""; candidate = None
             if hasattr(response, 'candidates') and response.candidates: candidate = response.candidates[0]
             if candidate and candidate.finish_reason == 1 and hasattr(candidate.content, 'parts') and candidate.content.parts: generated_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
             elif candidate and candidate.finish_reason != 0: reason = genai.types.Candidate.FinishReason(candidate.finish_reason).name; print(f"Warning: Response blocked: {reason}"); generated_text = f"[Blocked: {reason}]"
             elif hasattr(response, 'text'): generated_text = response.text
             elif hasattr(response, 'parts') and response.parts: generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
             if not generated_text and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: reason = genai.types.BlockReason(response.prompt_feedback.block_reason).name; print(f"Warning: Prompt blocked: {reason}"); generated_text = f"[Blocked: {reason}]"
             elif not generated_text and candidate is None and not hasattr(response, 'text') and not hasattr(response, 'parts'): print(f"Warning: No text in response: {response}")
             return CompletionResponse(text=generated_text)
         except Exception as e: print(f"Error calling generate_content: {e}"); raise
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse: return self._generate_sync(prompt, **kwargs)
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        print("Warning: Streaming not implemented."); response = self._generate_sync(prompt, **kwargs)
        def gen() -> CompletionResponseGen: yield CompletionResponse(text=response.text, delta=response.text)
        return gen()
    async def _agenerate_sync(self, prompt: str, **kwargs) -> CompletionResponse: return self._generate_sync(prompt, **kwargs)
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse: return await self._agenerate_sync(prompt, **kwargs)
    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        print("Warning: Async streaming not implemented."); response = await self._agenerate_sync(prompt, **kwargs)
        async def gen() -> AsyncGenerator[CompletionResponse, None]: yield CompletionResponse(text=response.text, delta=response.text)
        return gen()

# --- Định nghĩa Lớp Embedding Tùy chỉnh ---
class MyGeminiEmbedding(BaseEmbedding):
    model_name: str = "models/embedding-001"
    task_type: str = "RETRIEVAL_DOCUMENT"
    def __init__(self, model_name: str = "models/embedding-001", task_type: str = "RETRIEVAL_DOCUMENT", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        if self.model_name not in available_embedding_models:
            print(f"Warning: Embedding Model '{self.model_name}' not available. Trying fallback.")
            if available_embedding_models: self.model_name = available_embedding_models[0]; print(f"Switched to: {self.model_name}")
            else: raise ValueError("No Embedding Model available.")
        self.task_type = task_type; print(f"MyGeminiEmbedding initialized: {self.model_name}")
    def _get_embedding(self, text: str, current_task_type: str) -> List[float]:
        try:
            result = genai.embed_content(model=self.model_name, content=text, task_type=current_task_type)
            if 'embedding' in result and isinstance(result['embedding'], list): return result['embedding']
            else: print(f"Error: Invalid embedding format: {result}"); raise ValueError("Invalid format")
        except Exception as e: print(f"Error getting embedding: {e}"); raise
    def _get_text_embedding(self, text: str) -> List[float]: return self._get_embedding(text, "RETRIEVAL_QUERY")
    def _get_text_embedding_batch(self, texts: List[str], show_progress: bool = False, **kwargs: Any) -> List[List[float]]:
        # Bỏ tqdm để tránh lỗi với cache_resource
        return [self._get_embedding(text, "RETRIEVAL_DOCUMENT") for text in texts]
    def _get_query_embedding(self, query: str) -> List[float]: return self._get_embedding(query, "RETRIEVAL_QUERY")
    async def _aget_text_embedding(self, text: str) -> List[float]: return self._get_embedding(text, "RETRIEVAL_QUERY")
    async def _aget_text_embedding_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]: return [self._get_embedding(text, "RETRIEVAL_DOCUMENT") for text in texts]
    async def _aget_query_embedding(self, query: str) -> List[float]: return self._get_embedding(query, "RETRIEVAL_QUERY")

# --- Hàm Cache để Khởi tạo Models và Index ---
@st.cache_resource # Cache để không khởi tạo lại LLM/Embed/Index mỗi lần rerun
def load_resources():
    """Khởi tạo LLM, Embed Model, nạp dữ liệu và tạo Index."""
    print("--- Loading Resources (Cache Miss or First Run) ---") # In ra để biết khi nào cache chạy
    # Configure Settings
    target_llm_model, target_embed_model = None, None
    preferred_llms = ["models/gemini-1.5-flash-latest", "models/gemini-1.5-pro-latest", "models/gemini-pro"]
    preferred_embeds = ["models/embedding-001", "models/text-embedding-004"] # Thêm model mới của Google nếu có
    for pref in preferred_llms:
        if pref in available_llm_models: target_llm_model = pref; break
    if not target_llm_model and available_llm_models: target_llm_model = available_llm_models[0]
    for pref in preferred_embeds:
        if pref in available_embedding_models: target_embed_model = pref; break
    if not target_embed_model and available_embedding_models: target_embed_model = available_embedding_models[0]
    if not target_llm_model: raise ValueError("Không tìm thấy model LLM.")
    if not target_embed_model: raise ValueError("Không tìm thấy model Embedding.")

    print(f"Using LLM: {target_llm_model}"); print(f"Using Embedding Model: {target_embed_model}")
    llm = MyGeminiLLM(model_name=target_llm_model)
    embed_model = MyGeminiEmbedding(model_name=target_embed_model)
    Settings.llm = llm; Settings.embed_model = embed_model
    print("LlamaIndex Settings configured.")

    # Load Data and Create Index
    DATA_DIR = "data"; index = None; documents = []
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        print(f"Loading data from '{DATA_DIR}'...")
        try:
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            if documents:
                print(f"Loaded {len(documents)} documents. Creating index...")
                index = VectorStoreIndex.from_documents(documents)
                print("Index created.")
            else: print(f"No documents found in '{DATA_DIR}'.")
        except Exception as e: print(f"Error loading/indexing data: {e}")
    else: print(f"Directory '{DATA_DIR}' is empty or not found. Skipping RAG setup.")

    # Create Query Engine if index exists
    query_engine = index.as_query_engine() if index else None
    if query_engine: print("Query engine created.")
    else: print("Query engine not created (no index).")

    print("--- Finished Loading Resources ---")
    return query_engine, Settings.llm # Trả về engine và llm để dùng

# --- Khởi tạo và Lấy tài nguyên từ Cache ---
try:
    query_engine, llm_global = load_resources()
except Exception as e:
    st.error(f"Lỗi nghiêm trọng khi tải tài nguyên: {e}")
    st.stop()

# --- Giao diện Chat ---
st.divider()
if "messages" not in st.session_state:
    welcome = "Hỏi về dữ liệu trong 'data' nhé." if query_engine else "Trò chuyện với Gemini nhé (không có dữ liệu nền)."
    st.session_state.messages = [{"role": "assistant", "content": welcome}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty(); message_placeholder.markdown("🤔...")
        try:
            if query_engine:
                response = query_engine.query(prompt)
                full_response = str(response) if response is not None else "[No RAG response]"
                message_placeholder.markdown(full_response)
            elif llm_global: # Sử dụng llm đã lấy từ cache
                 st.info("Hỏi trực tiếp LLM...", icon="💬")
                 response = llm_global.complete(prompt)
                 full_response = response.text
                 message_placeholder.markdown(full_response)
            else: full_response = "Lỗi: Không có Query Engine/LLM."; message_placeholder.error(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            error_message = f"Lỗi: {e}"; st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            message_placeholder.markdown(error_message)