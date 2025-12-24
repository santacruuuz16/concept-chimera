"""
make_light_dict.py
Create a full word vector dictionary from the ChiVe model.
Loads ALL words for maximum vocabulary coverage.
Uses memory-mapped files for efficient loading.
"""

from gensim.models import KeyedVectors
import os

# Configuration
INPUT_MODEL = "chive-1.2-mc5/chive-1.2-mc5.txt"  # Text format from tar.gz
OUTPUT_FILE = "chimera_full.kv"

def main():
    print(f"Loading FULL model from {INPUT_MODEL}...")
    print("This will take several minutes for ~3.2 million words...")
    
    # Load ALL words (no limit parameter)
    full_model = KeyedVectors.load_word2vec_format(
        INPUT_MODEL, 
        binary=False
    )
    
    print(f"Model loaded. Total words: {len(full_model):,}")
    
    # Save the full model
    full_model.save(OUTPUT_FILE)
    
    # Check file sizes
    print(f"\nSaved to {OUTPUT_FILE}")
    
    total_size = 0
    for f in [OUTPUT_FILE, OUTPUT_FILE + ".vectors.npy"]:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            total_size += size_mb
            print(f"  {f}: {size_mb:.2f} MB")
    
    print(f"  Total: {total_size:.2f} MB")
    
    # Test the model with mmap (as it will be used in production)
    print("\n--- Quick Test (with mmap='r') ---")
    test_model = KeyedVectors.load(OUTPUT_FILE, mmap='r')
    
    test_words = ["王", "女", "男", "国", "東京"]
    for word in test_words:
        if word in test_model:
            similar = test_model.most_similar(word, topn=3)
            print(f"Words similar to '{word}': {[w for w, _ in similar]}")
        else:
            print(f"Test word '{word}' not in vocabulary")

if __name__ == "__main__":
    main()
