"""
Concept Chimera - Word Vector Mixing Lab
A "Mad Scientist Lab" themed app that mixes word vectors.
"""

import streamlit as st
from gensim.models import KeyedVectors
import os

# Page configuration
st.set_page_config(
    page_title="Concept Chimera 🧬",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Retro 8-bit / Cyberpunk" theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');

    /* Global styles */
    .stApp {
        background-color: #0e1117;
        font-family: 'DotGothic16', sans-serif !important;
        color: #00ff41 !important;
    }

    * {
        font-family: 'DotGothic16', sans-serif !important;
    }

    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        color: #ff007f;
        text-shadow: 3px 3px 0px #000, 6px 6px 0px #00ff41;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    
    .subtitle {
        text-align: center;
        color: #00ff41;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        background: rgba(0, 255, 65, 0.1);
        padding: 10px;
        border: 1px dashed #00ff41;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #000 !important;
        border: 2px solid #00ff41 !important;
        color: #00ff41 !important;
        font-size: 1.2rem !important;
        padding: 0.75rem !important;
        border-radius: 0px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ff007f !important;
        box-shadow: 0 0 10px #ff007f !important;
    }

    .stTextInput label {
        color: #00ff41 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #000 !important;
        color: #00ff41 !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        border: 4px solid #00ff41 !important;
        border-radius: 0px !important;
        width: 100% !important;
        transition: none !important;
        text-transform: uppercase;
        box-shadow: 4px 4px 0px #00ff41;
    }
    
    .stButton > button:hover {
        background-color: #00ff41 !important;
        color: #000 !important;
        box-shadow: none !important;
    }

    .stButton > button:active {
        transform: translateY(2px) translateX(2px) !important;
        box-shadow: 2px 2px 0px #00ff41 !important;
    }
    
    /* Result box */
    .result-box {
        background-color: #000;
        border: 4px double #00ff41;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        position: relative;
    }
    
    .result-word {
        font-size: 3.5rem;
        font-weight: bold;
        color: #ff007f;
        text-shadow: 2px 2px 0px #fff;
    }
    
    .formula {
        font-size: 1.5rem;
        color: #00ff41;
        margin-bottom: 1rem;
    }
    
    /* Warning styling */
    .warning-box {
        background-color: #200;
        border: 2px solid #ff0000;
        padding: 1.5rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .warning-text {
        color: #ff0000;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    /* CRT Scanline Effect */
    body::before {
        content: " ";
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
        z-index: 9999;
        background-size: 100% 4px, 3px 100%;
        pointer-events: none;
    }

    /* Lab equipment decorations (unused but kept for structure) */
    .lab-header {
        display: flex;
        justify-content: center;
        gap: 1rem;
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #00ff41;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the word vector model with memory mapping for efficiency."""
    model_path = "chimera_full.kv"
    npy_path = model_path + ".vectors.npy"
    parts_prefix = npy_path + ".part-"
    
    # Check if we need to reconstruct the large .npy file from split parts
    if not os.path.exists(npy_path):
        import glob
        parts = sorted(glob.glob(f"{parts_prefix}*"))
        if parts:
            with st.spinner("🔧 Reconstructing gene pool from fragments..."):
                with open(npy_path, "wb") as outfile:
                    for part in parts:
                        with open(part, "rb") as infile:
                            outfile.write(infile.read())
                st.success("✅ Gene pool reconstruction complete.")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
        
    # Use mmap='r' to prevent memory overflow on Streamlit Cloud
    return KeyedVectors.load(model_path, mmap='r')


def find_chimera(model, word_a: str, word_b: str) -> tuple[str | None, list]:
    """
    Mix two word vectors and find the closest result.
    Returns (result_word, similar_words_list) or (None, []) if error.
    """
    # Check if words exist in vocabulary
    if word_a not in model:
        return None, []
    if word_b not in model:
        return None, []
    
    # Vector arithmetic: A + B
    try:
        # Find most similar words to the sum of vectors
        # Exclude the input words from results
        result = model.most_similar(
            positive=[word_a, word_b],
            topn=5
        )
        
        # Filter out input words and get the top result
        filtered = [(word, score) for word, score in result 
                    if word != word_a and word != word_b]
        
        if filtered:
            return filtered[0][0], filtered[:3]
        else:
            return None, []
    except Exception:
        return None, []


def main():
    # Header
    st.markdown('<div class="lab-header">SYSTEM READY > _</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">CONCEPTS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">RETRO-SYNTHESIS INTERFACE v1.0.4<br>[ 概念融合プロトコル実行中 ]</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Input section
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        word_a = st.text_input("🧬 DNA Sample A", placeholder="例: 王", key="word_a")
    
    with col2:
        st.markdown("<div style='text-align: center; font-size: 2rem; color: #8b5cf6; padding-top: 1.8rem;'>＋</div>", unsafe_allow_html=True)
    
    with col3:
        word_b = st.text_input("🧬 DNA Sample B", placeholder="例: 女", key="word_b")
    
    # Mix button
    st.markdown("<br>", unsafe_allow_html=True)
    mix_button = st.button("⚡ FUSE CONCEPTS ⚡")
    
    # Process
    if mix_button and word_a and word_b:
        # Show loading animation
        with st.spinner("🔮 Synthesizing new concept..."):
            result_word, similar_words = find_chimera(model, word_a.strip(), word_b.strip())
        
        if result_word:
            # Success - show result
            st.markdown(f"""
            <div class="result-box">
                <div class="formula">{word_a} ＋ {word_b} ＝</div>
                <div class="result-word">✨ {result_word} ✨</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show other similar results
            if len(similar_words) > 1:
                st.markdown("**🔬 Other possible mutations:**")
                for word, score in similar_words[1:]:
                    st.markdown(f"- {word} (similarity: {score:.3f})")
        else:
            # Error - word not in dictionary
            st.markdown("""
            <div class="warning-box">
                <div class="warning-text">⚠️ Invalid DNA Sequence ⚠️</div>
                <p style="color: #aa6666;">One or both words are not in the gene pool.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with stats
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🧬 Gene Pool: {len(model):,} concepts | Powered by ChiVe Word Vectors
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
