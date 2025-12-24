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

# Custom CSS for "Mad Scientist Lab" theme
st.markdown("""
<style>
    /* Dark theme with green/purple accents */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a0a2e 50%, #0a1a0a 100%);
    }
    
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00ff88, #8b5cf6, #00ff88);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
        margin-bottom: 0.5rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #1a1a2e !important;
        border: 2px solid #8b5cf6 !important;
        color: #00ff88 !important;
        font-size: 1.2rem !important;
        padding: 0.75rem !important;
        border-radius: 10px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #8b5cf6, #00ff88) !important;
        color: #000 !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        border: none !important;
        border-radius: 25px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.5) !important;
    }
    
    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1a0a2e, #0a2a1a);
        border: 2px solid #00ff88;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.2);
    }
    
    .result-word {
        font-size: 3rem;
        font-weight: bold;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.8);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .formula {
        font-size: 1.5rem;
        color: #8b5cf6;
        margin-bottom: 1rem;
        font-family: monospace;
    }
    
    /* Warning styling */
    .warning-box {
        background: linear-gradient(135deg, #2e1a0a, #2e0a0a);
        border: 2px solid #ff4444;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .warning-text {
        color: #ff4444;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    /* Lab equipment decorations */
    .lab-header {
        display: flex;
        justify-content: center;
        gap: 1rem;
        font-size: 2rem;
        margin-bottom: 1rem;
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
    st.markdown('<div class="lab-header">🧪 ⚗️ 🔬</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">CONCEPT CHIMERA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">~ 概念融合研究所 ~<br>Mix word DNA to create new concepts</p>', unsafe_allow_html=True)
    
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
