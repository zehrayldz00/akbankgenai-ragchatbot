# chatbot.py — FINAL (buttons, no gr.Examples)
import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import gradio as gr
import chromadb
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts

cini_theme = Base(
    primary_hue=colors.sky,     # mavi
    secondary_hue=colors.rose,  # soft kırmızı
    neutral_hue=colors.slate,
    font=fonts.GoogleFont("Poppins")
)

try:
    from chromadb.config import Settings
except Exception:
    Settings = None

load_dotenv()

API_KEY          = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "istanbulguide_collection")
PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_persist")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
TOP_K            = int(os.getenv("TOP_K", "8"))
BAD_DIST_THRESH  = float(os.getenv("BAD_DIST_THRESH", "999"))  # ilk testte kapalı

if not API_KEY:
    raise SystemExit("GEMINI_API_KEY missing in .env")

print("Loading embed model:", EMBED_MODEL_NAME)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ---- Chroma ----
def create_chroma_client(persist_path: str):
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=persist_path)
    if Settings is None:
        raise SystemExit("Chroma version unsupported.")
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_path)
    return chromadb.Client(settings=settings)

client = create_chroma_client(PERSIST_DIR)

def get_existing_collection(client, name: str):
    if hasattr(client, "get_collection"):
        return client.get_collection(name)
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(name)
    raise SystemExit("Collection not found. Run reindex_persist.py first.")

collection = get_existing_collection(client, COLLECTION_NAME)

# ---- Retrieval ----
def retrieve_top_k(question, k=TOP_K, bad_dist_threshold=BAD_DIST_THRESH):
    q_emb = embed_model.encode([question], convert_to_numpy=True)[0].tolist()
    try:
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
    except TypeError:
        res = collection.query(query_embeddings=[q_emb], n_results=k)

    docs  = (res.get("documents")  or [[]])[0]
    metas = (res.get("metadatas")  or [[]])[0]
    dists = (res.get("distances")  or [[]])[0]

    if bad_dist_threshold is not None and dists and len(dists) > 0:
        best_dist = dists[0]
        if best_dist is not None and best_dist > bad_dist_threshold:
            return [], []
    return docs, metas

def build_prompt_rag(question, docs, metas):
    context = "\n\n---\n\n".join((d or "")[:1200].replace("\n", " ").strip() for d in docs)
    return f"""You are a friendly, factual Istanbul travel guide.
Use ONLY the context below to answer. If the answer is not in the context, say:
"I don't know — the guide does not contain that information."
Reply in clear, natural English.

Context:
{context}

User question: {question}
"""

def build_prompt_chat(message):
    return f"""You are a friendly Istanbul travel concierge chatting with a tourist.
Keep it light, helpful, and engaging. Reply in natural English. Avoid specific factual claims
(prices/hours/history) unless user asks and you are given context.

Tourist: {message}
Assistant:"""

# ---- LLM ----
def _extract_text(resp):
    if not resp: return ""
    t = getattr(resp, "text", None)
    if t: return t
    try:
        return "\n".join(
            p.text for c in getattr(resp, "candidates", []) for p in getattr(c.content, "parts", []) if hasattr(p, "text")
        )
    except Exception:
        return str(resp) if resp else ""

def call_llm(prompt_text):
    # yeni SDK
    try:
        import google.genai as genai
        client = genai.Client(api_key=API_KEY)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_text)
        txt = _extract_text(resp)
        if txt: return txt
    except Exception:
        pass
    # eski SDK
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt_text)
        txt = _extract_text(resp)
        if txt: return txt
    except Exception:
        pass
    return ""

# ---- Intent ----
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|selam|merhaba|sa|hey there)\s*[!?\.]*\s*$", re.I)
INFO_KEYWORDS = re.compile(
    r"(tell me about|what is|who built|where is|history|facts?|opening hours|tickets?|price|"
    r"how (to|do i) get|directions|best|top|museum|mosque|palace|bazaar|airport|"
    r"grand bazaar|hagia sophia|ayasofya|topkap[iı]|sultanahmet|dolmabah[çc]e)", re.I
)

def detect_intent(msg):
    if GREETING_RE.match(msg): return "chat"
    if INFO_KEYWORDS.search(msg): return "rag"
    return "rag"

def _looks_empty(text): return not text or not text.strip()

# ---- Chat ----
def chat_function(message):
    if not message or not message.strip():
        return "Please type a message."
    intent = detect_intent(message)

    if intent == "chat":
        resp = call_llm(build_prompt_chat(message))
        return resp.strip() if not _looks_empty(resp) else "Hi! How can I help with your Istanbul trip today?"

    docs, metas = retrieve_top_k(message)
    if not docs or all(not d for d in docs):
        return ("I don't have that in the guide right now. "
                "Tell me what you’re looking for—history, how to get there, or ticket tips—and I’ll try to assist!")
    resp = call_llm(build_prompt_rag(message, docs, metas))
    return resp.strip() if not _looks_empty(resp) else "Hmm, I couldn’t find a helpful answer. Try rephrasing."


# ---- UI ----
with gr.Blocks(theme=cini_theme, title="Istanbul Travel Assistant") as demo:
    # HEADER (küçük ve yukarıdan başlar)
    gr.HTML("""
    <style>
    #main_column {
        display: flex;
        flex-direction: column;
        justify-content: center;   /* dikey ortala */
        align-items: center;       /* yatay ortala */
        padding-top: 0;
        margin-top: 0;
    }
    </style>
    """)


    # MERKEZDE DARALMIŞ BLOK
with gr.Blocks(theme=cini_theme, title="Istanbul Travel Assistant") as demo:
    # CSS hizalama
    gr.HTML("""
    <style>
    #main_column {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;  /* en tepeden başla */
        align-items: center;           /* yatay ortala */
        min-height: 100vh;
        padding-top: 40px;
    }
    </style>
    """)

    # Başlık ve açıklama
    with gr.Column(elem_id="main_column", scale=1, min_width=600):
        gr.HTML("<h2 style='margin-bottom:10px;'>🧭 Istanbul Travel Assistant</h2>")

        user_input = gr.Textbox(
            label="Your question",
            placeholder="Ask me anything about Istanbul…",
            lines=4
        )

        with gr.Row(equal_height=True):
            ask_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Row(equal_height=True):
            t1 = gr.Button("🏛 Hagia Sophia", variant="secondary")
            t2 = gr.Button("🕌 Blue Mosque", variant="secondary")
            t3 = gr.Button("🛍 Grand Bazaar", variant="secondary")
            t4 = gr.Button("🏰 Dolmabahçe", variant="secondary")

        answer_box = gr.Textbox(
            label="Answer",
            show_copy_button=True,
            lines=10
        )

    # Event bağlantıları
    ask_btn.click(fn=chat_function, inputs=user_input, outputs=answer_box)
    user_input.submit(fn=chat_function, inputs=user_input, outputs=answer_box)
    clear_btn.click(fn=lambda: ("", ""), outputs=[user_input, answer_box])

    # Template butonlar
    t1.click(lambda: "Tell me about Hagia Sophia.", inputs=None, outputs=user_input)
    t2.click(lambda: "Tell me about the Blue Mosque.", inputs=None, outputs=user_input)
    t3.click(lambda: "What is the Grand Bazaar known for?", inputs=None, outputs=user_input)
    t4.click(lambda: "Tell me about Dolmabahçe Palace.", inputs=None, outputs=user_input)

    # Footer
    gr.HTML("<p style='text-align:center;opacity:0.7;margin-top:16px;'>Made with ❤️ in Istanbul</p>")

    
if __name__ == "__main__":
    demo.launch(share=True)
