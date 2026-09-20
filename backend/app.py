"""
Cortex — general-purpose RAG chatbot
Streamlit front end. Run with: streamlit run app.py
"""

import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

from rag_engine import RagEngine

load_dotenv()

st.set_page_config(page_title="Cortex — chat with your documents", page_icon="◆", layout="wide")

# A little custom CSS for the parts config.toml can't reach — the italic
# serif wordmark, and hiding Streamlit's default "Made with Streamlit"
# footer so the page reads as its own product.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,480;1,400&display=swap');
    .cortex-title {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 500;
        font-size: 1.5rem;
        color: #E7ECF2;
        margin-bottom: 0.25rem;
    }
    .cortex-title span { color: #00D9B5; }
    .cortex-subtitle {
        font-family: 'Fraunces', serif;
        font-style: italic;
        color: #8CA0B3;
        font-size: 1.02rem;
        margin-bottom: 1.2rem;
    }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    return RagEngine()


engine = get_engine()


# ------------------------------------------------------------------------ #
# Conversation state
# st.session_state.conversations: { id: {title, history, session_id, file_names} }
# ------------------------------------------------------------------------ #

def _new_conversation():
    conv_id = str(uuid.uuid4())
    session_id = engine.new_session()
    return conv_id, {
        "title": "New chat",
        "history": [],
        "session_id": session_id,
        "file_names": [],
        "order": len(st.session_state.get("conversations", {})),
    }


if "conversations" not in st.session_state:
    conv_id, conv = _new_conversation()
    st.session_state.conversations = {conv_id: conv}
    st.session_state.current_id = conv_id

conversations = st.session_state.conversations
current_id = st.session_state.current_id
current = conversations[current_id]


# ------------------------------------------------------------------------ #
# Sidebar: new chat, previous chats, upload
# ------------------------------------------------------------------------ #
with st.sidebar:
    st.markdown('<div class="cortex-title">Cortex<span>.</span></div>', unsafe_allow_html=True)

    if st.button("+ New chat", type="primary", use_container_width=True):
        conv_id, conv = _new_conversation()
        st.session_state.conversations[conv_id] = conv
        st.session_state.current_id = conv_id
        st.rerun()

    st.caption("Previous chats")
    for cid, conv in sorted(
        conversations.items(), key=lambda kv: kv[1]["order"], reverse=True
    ):
        marker = "●" if cid == current_id else "○"
        col_select, col_delete = st.columns([5, 1])
        with col_select:
            if st.button(f"{marker} {conv['title']}", key=f"conv_{cid}", use_container_width=True):
                st.session_state.current_id = cid
                st.rerun()
        with col_delete:
            if st.button("🗑", key=f"del_{cid}", help="Delete this chat"):
                engine.delete_session(conv["session_id"])
                del st.session_state.conversations[cid]

                if not st.session_state.conversations:
                    # always keep at least one chat open
                    new_id, new_conv = _new_conversation()
                    st.session_state.conversations[new_id] = new_conv
                    st.session_state.current_id = new_id
                elif cid == current_id:
                    # deleted the active chat — fall back to the most recent one left
                    remaining = sorted(
                        st.session_state.conversations.items(),
                        key=lambda kv: kv[1]["order"],
                        reverse=True,
                    )
                    st.session_state.current_id = remaining[0][0]

                st.rerun()

    st.divider()

    # Keying the uploader by the conversation id means switching chats
    # naturally gives you a fresh, empty upload widget — no leftover file
    # from a different conversation showing here.
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{current_id}",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name in current["file_names"]:
                continue  # already indexed this run
            suffix = os.path.splitext(uf.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name
            try:
                engine.add_document(current["session_id"], tmp_path, display_name=uf.name)
                current["file_names"].append(uf.name)
            except RuntimeError as e:
                st.error(f"Couldn't index {uf.name}: {e}")
            finally:
                os.unlink(tmp_path)

    if current["file_names"]:
        st.caption(f"Indexed: {', '.join(current['file_names'])}")
    else:
        st.caption("No documents indexed yet.")


# ------------------------------------------------------------------------ #
# Main chat area
# ------------------------------------------------------------------------ #
st.markdown(
    '<div class="cortex-subtitle">Ask questions grounded only in the documents '
    "you've uploaded in this chat. Every answer cites the passage it came from.</div>",
    unsafe_allow_html=True,
)

for turn in current["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if prompt := st.chat_input("Ask something about the documents you uploaded…"):
    current["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, sources = engine.query(current["session_id"], prompt, current["history"][:-1])
            if sources:
                def _cite_line(s):
                    if s.score >= 0.999:
                        return f"[{s.index}] {s.file_name}"
                    return f"[{s.index}] {s.file_name} — match {s.score:.0%}"

                answer += "\n\n---\n**Sources**\n" + "\n".join(_cite_line(s) for s in sources)
        st.markdown(answer)

    current["history"].append({"role": "assistant", "content": answer})
    if current["title"] == "New chat":
        current["title"] = prompt.strip()[:40] + ("…" if len(prompt.strip()) > 40 else "")
    st.rerun()
