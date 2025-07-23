"""
Advanced Text Preprocessing Techniques Animation
This script demonstrates advanced preprocessing methods like n-grams, TF-IDF, and normalization
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from collections import Counter, defaultdict
import math

class AdvancedPreprocessingAnimator:
    def __init__(self, text_corpus):
        self.text_corpus = text_corpus
        self.documents = [doc.strip() for doc in text_corpus.split('.') if doc.strip()]
        
        # Preprocessing results storage
        self.processing_results = {}
        
        self.fig = plt.figure(figsize=(18, 12))
        self.setup_subplots()
        
        # Perform all advanced preprocessing
        self.perform_advanced_preprocessing()
        
    def setup_subplots(self):
        """Setup subplot layout"""
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], width_ratios=[1, 1])
        
        self.ax_input = self.fig.add_subplot(gs[0, :])
        self.ax_left = self.fig.add_subplot(gs[1, 0])
        self.ax_right = self.fig.add_subplot(gs[1, 1])
        self.ax_output = self.fig.add_subplot(gs[2, :])
    
    def perform_advanced_preprocessing(self):
        """Perform advanced preprocessing techniques"""
        # Clean and tokenize documents
        cleaned_docs = []
        for doc in self.documents:
            # Basic cleaning
            clean_doc = doc.lower().strip()
            clean_doc = ''.join(c for c in clean_doc if c.isalnum() or c.isspace())
            tokens = clean_doc.split()
            cleaned_docs.append(tokens)
        
        self.cleaned_docs = cleaned_docs
        
        # Generate n-grams
        self.unigrams = self.generate_ngrams(cleaned_docs, 1)
        self.bigrams = self.generate_ngrams(cleaned_docs, 2)
        self.trigrams = self.generate_ngrams(cleaned_docs, 3)
        
        # Calculate TF-IDF
        self.tf_scores = self.calculate_tf(cleaned_docs)
        self.idf_scores = self.calculate_idf(cleaned_docs)
        self.tfidf_scores = self.calculate_tfidf()
        
        # Text normalization techniques
        self.normalization_examples = self.create_normalization_examples()
        
    def generate_ngrams(self, docs, n):
        """Generate n-grams from documents"""
        ngrams = []
        for doc in docs:
            doc_ngrams = []
            for i in range(len(doc) - n + 1):
                ngram = ' '.join(doc[i:i+n])
                doc_ngrams.append(ngram)
            ngrams.append(doc_ngrams)
        return ngrams
    
    def calculate_tf(self, docs):
        """Calculate Term Frequency for each document"""
        tf_scores = []
        for doc in docs:
            word_count = len(doc)
            tf_dict = {}
            for word in doc:
                tf_dict[word] = tf_dict.get(word, 0) + 1
            
            # Normalize by document length
            for word in tf_dict:
                tf_dict[word] = tf_dict[word] / word_count
            
            tf_scores.append(tf_dict)
        
        return tf_scores
    
    def calculate_idf(self, docs):
        """Calculate Inverse Document Frequency"""
        # Get all unique words
        all_words = set()
        for doc in docs:
            all_words.update(doc)
        
        idf_dict = {}
        total_docs = len(docs)
        
        for word in all_words:
            # Count documents containing this word
            containing_docs = sum(1 for doc in docs if word in doc)
            # Calculate IDF
            idf_dict[word] = math.log(total_docs / containing_docs)
        
        return idf_dict
    
    def calculate_tfidf(self):
        """Calculate TF-IDF scores"""
        tfidf_scores = []
        for tf_dict in self.tf_scores:
            tfidf_dict = {}
            for word, tf_score in tf_dict.items():
                idf_score = self.idf_scores[word]
                tfidf_dict[word] = tf_score * idf_score
            tfidf_scores.append(tfidf_dict)
        
        return tfidf_scores
    
    def create_normalization_examples(self):
        """Create examples of text normalization techniques"""
        return {
            'contractions': {
                "don't": "do not",
                "won't": "will not", 
                "can't": "cannot",
                "I'm": "I am",
                "you're": "you are"
            },
            'abbreviations': {
                "NLP": "Natural Language Processing",
                "AI": "Artificial Intelligence",
                "ML": "Machine Learning",
                "USA": "United States of America"
            },
            'numbers': {
                "123": "one hundred twenty three",
                "1st": "first",
                "2nd": "second",
                "50%": "fifty percent"
            },
            'unicode': {
                "café": "cafe",
                "naïve": "naive",
                "résumé": "resume"
            }
        }
    
    def animate_advanced_preprocessing(self, frame):
        """Animate advanced preprocessing techniques"""
        # Clear all axes
        for ax in [self.ax_input, self.ax_left, self.ax_right, self.ax_output]:
            ax.clear()
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
        
        techniques = [
            "N-gram Generation",
            "TF-IDF Calculation", 
            "Text Normalization",
            "Feature Extraction Summary"
        ]
        
        if frame < len(techniques):
            technique = techniques[frame]
            self.show_technique_header(technique)
            
            if technique == "N-gram Generation":
                self.show_ngram_generation()
            elif technique == "TF-IDF Calculation":
                self.show_tfidf_calculation()
            elif technique == "Text Normalization":
                self.show_text_normalization()
            elif technique == "Feature Extraction Summary":
                self.show_feature_summary()
        
        return []
    
    def show_technique_header(self, technique):
        """Show current technique header"""
        self.fig.suptitle(f"🔬 Advanced Text Preprocessing: {technique}", 
                         fontsize=16, fontweight='bold', y=0.95)
    
    def show_ngram_generation(self):
        """Visualize n-gram generation"""
        self.ax_input.text(5, 8, "N-gram Generation", ha='center', fontsize=14, fontweight='bold')
        self.ax_input.text(5, 7, "Creating sequences of N consecutive words", 
                          ha='center', fontsize=12, style='italic')
        
        # Show example sentence
        example_doc = self.cleaned_docs[0] if self.cleaned_docs else ["natural", "language", "processing", "is", "amazing"]
        self.ax_input.text(5, 5.5, f"Example: {' '.join(example_doc)}", 
                          ha='center', fontsize=11,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Left panel: Show unigrams and bigrams
        self.ax_left.text(5, 9, "Unigrams (1-grams)", ha='center', fontsize=12, fontweight='bold', color='blue')
        unigrams = example_doc
        y_pos = 8
        for i, unigram in enumerate(unigrams[:6]):
            self.ax_left.text(5, y_pos - i*0.5, f"'{unigram}'", ha='center', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcyan"))
        
        # Right panel: Show bigrams and trigrams
        self.ax_right.text(5, 9, "Bigrams (2-grams)", ha='center', fontsize=12, fontweight='bold', color='green')
        bigrams = [' '.join(example_doc[i:i+2]) for i in range(len(example_doc)-1)]
        y_pos = 8
        for i, bigram in enumerate(bigrams[:5]):
            self.ax_right.text(5, y_pos - i*0.5, f"'{bigram}'", ha='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
        
        # Show trigrams in output
        self.ax_output.text(5, 8, "Trigrams (3-grams)", ha='center', fontsize=12, fontweight='bold', color='red')
        trigrams = [' '.join(example_doc[i:i+3]) for i in range(len(example_doc)-2)]
        trigram_text = " | ".join(f"'{tg}'" for tg in trigrams[:3])
        self.ax_output.text(5, 6.5, trigram_text, ha='center', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        # Show statistics
        self.ax_output.text(5, 4, f"Generated: {len(unigrams)} unigrams, {len(bigrams)} bigrams, {len(trigrams)} trigrams", 
                          ha='center', fontsize=11, fontweight='bold')
    
    def show_tfidf_calculation(self):
        """Visualize TF-IDF calculation"""
        self.ax_input.text(5, 8, "TF-IDF (Term Frequency - Inverse Document Frequency)", 
                          ha='center', fontsize=14, fontweight='bold')
        self.ax_input.text(5, 7, "Measures word importance across document collection", 
                          ha='center', fontsize=12, style='italic')
        
        # Left panel: Show TF calculation
        self.ax_left.text(5, 9, "Term Frequency (TF)", ha='center', fontsize=12, fontweight='bold', color='blue')
        self.ax_left.text(5, 8.5, "TF = (word count) / (total words)", ha='center', fontsize=10, style='italic')
        
        if self.tf_scores:
            # Show TF for first document
            tf_dict = self.tf_scores[0]
            top_tf = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            
            y_pos = 7.5
            for word, tf_score in top_tf:
                self.ax_left.text(5, y_pos, f"'{word}': {tf_score:.3f}", ha='center', fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue"))
                y_pos -= 0.6
        
        # Right panel: Show IDF calculation
        self.ax_right.text(5, 9, "Inverse Document Frequency (IDF)", ha='center', fontsize=12, fontweight='bold', color='green')
        self.ax_right.text(5, 8.5, "IDF = log(total docs / docs with word)", ha='center', fontsize=10, style='italic')
        
        if self.idf_scores:
            # Show IDF scores
            top_idf = sorted(self.idf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            y_pos = 7.5
            for word, idf_score in top_idf:
                self.ax_right.text(5, y_pos, f"'{word}': {idf_score:.3f}", ha='center', fontsize=10,
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
                y_pos -= 0.6
        
        # Show TF-IDF results
        self.ax_output.text(5, 8, "TF-IDF = TF × IDF", ha='center', fontsize=12, fontweight='bold', color='red')
        
        if self.tfidf_scores:
            tfidf_dict = self.tfidf_scores[0]
            top_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:4]
            
            tfidf_text = " | ".join(f"'{word}': {score:.3f}" for word, score in top_tfidf)
            self.ax_output.text(5, 6.5, tfidf_text, ha='center', fontsize=10,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            
            self.ax_output.text(5, 4, "Higher TF-IDF = More important to this document", 
                              ha='center', fontsize=11, fontweight='bold')
    
    def show_text_normalization(self):
        """Visualize text normalization techniques"""
        self.ax_input.text(5, 8, "Text Normalization Techniques", ha='center', fontsize=14, fontweight='bold')
        self.ax_input.text(5, 7, "Standardizing text variations and formats", 
                          ha='center', fontsize=12, style='italic')
        
        # Left panel: Contractions and abbreviations
        self.ax_left.text(5, 9, "Contractions & Abbreviations", ha='center', fontsize=12, fontweight='bold', color='blue')
        
        examples = list(self.normalization_examples['contractions'].items())[:3] + \
                  list(self.normalization_examples['abbreviations'].items())[:2]
        
        y_pos = 8
        for original, normalized in examples:
            self.ax_left.text(2.5, y_pos, original, ha='center', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral"))
            self.ax_left.text(5, y_pos, "→", ha='center', fontsize=12, fontweight='bold')
            self.ax_left.text(7.5, y_pos, normalized, ha='center', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
            y_pos -= 0.8
        
        # Right panel: Numbers and Unicode
        self.ax_right.text(5, 9, "Numbers & Unicode", ha='center', fontsize=12, fontweight='bold', color='green')
        
        examples = list(self.normalization_examples['numbers'].items())[:2] + \
                  list(self.normalization_examples['unicode'].items())[:3]
        
        y_pos = 8
        for original, normalized in examples:
            self.ax_right.text(2.5, y_pos, original, ha='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral"))
            self.ax_right.text(5, y_pos, "→", ha='center', fontsize=12, fontweight='bold')
            self.ax_right.text(7.5, y_pos, normalized, ha='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
            y_pos -= 0.8
        
        # Show benefits
        self.ax_output.text(5, 7, "Benefits of Normalization:", ha='center', fontsize=12, fontweight='bold')
        benefits = [
            "• Reduces vocabulary size",
            "• Improves model consistency", 
            "• Handles text variations",
            "• Better feature matching"
        ]
        
        y_pos = 6
        for benefit in benefits:
            self.ax_output.text(5, y_pos, benefit, ha='center', fontsize=10)
            y_pos -= 0.5
    
    def show_feature_summary(self):
        """Show summary of all preprocessing features"""
        self.ax_input.text(5, 8, "Feature Extraction Summary", ha='center', fontsize=14, fontweight='bold')
        self.ax_input.text(5, 7, "Complete preprocessing pipeline results", 
                          ha='center', fontsize=12, style='italic')
        
        # Left panel: Feature counts
        self.ax_left.text(5, 9, "Feature Statistics", ha='center', fontsize=12, fontweight='bold', color='blue')
        
        # Calculate statistics
        total_unigrams = sum(len(doc_unigrams) for doc_unigrams in self.unigrams)
        total_bigrams = sum(len(doc_bigrams) for doc_bigrams in self.bigrams)
        total_trigrams = sum(len(doc_trigrams) for doc_trigrams in self.trigrams)
        unique_words = len(self.idf_scores) if self.idf_scores else 0
        
        stats = [
            f"Documents: {len(self.documents)}",
            f"Unique words: {unique_words}",
            f"Total unigrams: {total_unigrams}",
            f"Total bigrams: {total_bigrams}",
            f"Total trigrams: {total_trigrams}"
        ]
        
        y_pos = 8
        for stat in stats:
            self.ax_left.text(5, y_pos, stat, ha='center', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue"))
            y_pos -= 0.7
        
        # Right panel: Processing pipeline
        self.ax_right.text(5, 9, "Processing Pipeline", ha='center', fontsize=12, fontweight='bold', color='green')
        
        pipeline_steps = [
            "1. Text cleaning",
            "2. Tokenization", 
            "3. N-gram generation",
            "4. TF-IDF calculation",
            "5. Normalization",
            "6. Feature extraction"
        ]
        
        y_pos = 8
        for step in pipeline_steps:
            self.ax_right.text(5, y_pos, step, ha='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
            y_pos -= 0.7
        
        # Show final message
        self.ax_output.text(5, 6, "🎉 Text Preprocessing Complete!", ha='center', fontsize=14, fontweight='bold', color='green')
        self.ax_output.text(5, 5, "Text is now ready for machine learning models", 
                          ha='center', fontsize=12, style='italic')
        
        # Show next steps
        self.ax_output.text(5, 3.5, "Next Steps:", ha='center', fontsize=12, fontweight='bold')
        next_steps = "Classification • Clustering • Sentiment Analysis • Topic Modeling"
        self.ax_output.text(5, 2.5, next_steps, ha='center', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

def demonstrate_advanced_preprocessing():
    """Run the advanced preprocessing animation demo"""
    sample_corpus = """
    Natural Language Processing is amazing and powerful.
    Machine Learning algorithms can process text data efficiently.
    NLP techniques include tokenization, stemming, and TF-IDF calculation.
    Advanced preprocessing improves model performance significantly.
    Text normalization handles contractions like don't and won't.
    """
    
    print("🔬 Starting Advanced Text Preprocessing Animation Demo!")
    print("=" * 70)
    
    animator = AdvancedPreprocessingAnimator(sample_corpus)
    
    # Create animation
    anim = animation.FuncAnimation(
        animator.fig,
        animator.animate_advanced_preprocessing,
        frames=4,
        interval=5000,  # 5 seconds per frame
        repeat=True,
        blit=False
    )
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nAdvanced Preprocessing Results:")
    print("-" * 40)
    
    print(f"Documents processed: {len(animator.documents)}")
    print(f"Unique words: {len(animator.idf_scores)}")
    
    if animator.tfidf_scores:
        print("\nTop TF-IDF terms (first document):")
        tfidf_dict = animator.tfidf_scores[0]
        top_terms = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        for term, score in top_terms:
            print(f"  {term}: {score:.4f}")
    
    print("\nN-gram examples:")
    if animator.bigrams:
        print(f"  Bigrams: {animator.bigrams[0][:3]}")
    if animator.trigrams:
        print(f"  Trigrams: {animator.trigrams[0][:2]}")

def explain_advanced_concepts():
    """Explain advanced preprocessing concepts"""
    print("\n" + "="*70)
    print("🔬 ADVANCED TEXT PREPROCESSING CONCEPTS")
    print("="*70)
    print()
    print("🔢 N-GRAMS:")
    print("   • Unigrams: Individual words ('natural', 'language')")
    print("   • Bigrams: Two consecutive words ('natural language')")
    print("   • Trigrams: Three consecutive words ('natural language processing')")
    print("   • Captures word order and context")
    print("   • Used in language modeling and feature extraction")
    print()
    print("📊 TF-IDF (Term Frequency - Inverse Document Frequency):")
    print("   • TF: How often a word appears in a document")
    print("   • IDF: How rare a word is across all documents")
    print("   • TF-IDF = TF × IDF")
    print("   • Higher score = more important to that document")
    print("   • Reduces impact of common words")
    print()
    print("🔄 TEXT NORMALIZATION:")
    print("   • Contractions: don't → do not")
    print("   • Abbreviations: NLP → Natural Language Processing")
    print("   • Numbers: 123 → one hundred twenty three")
    print("   • Unicode: café → cafe")
    print("   • Standardizes text variations")
    print()
    print("🎯 FEATURE EXTRACTION:")
    print("   • Converts text to numerical features")
    print("   • Bag of Words (BoW)")
    print("   • TF-IDF vectors")
    print("   • N-gram features")
    print("   • Word embeddings")
    print()
    print("⚡ ADVANCED TECHNIQUES:")
    print("   • Dependency parsing")
    print("   • Named entity recognition")
    print("   • Part-of-speech tagging")
    print("   • Semantic role labeling")
    print("   • Coreference resolution")
    print()
    print("💡 BEST PRACTICES:")
    print("   • Choose techniques based on your task")
    print("   • Balance complexity vs. performance")
    print("   • Validate preprocessing impact")
    print("   • Consider domain-specific requirements")
    print("   • Document your preprocessing pipeline")
    print("="*70)

if __name__ == "__main__":
    explain_advanced_concepts()
    demonstrate_advanced_preprocessing()
