"""
Animated Word Frequency Analysis
This script shows how word frequencies change as we process more text
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
import numpy as np

class WordFrequencyAnimator:
    def __init__(self, text_corpus):
        self.text_corpus = text_corpus
        self.sentences = [sentence.strip() for sentence in text_corpus.split('.') if sentence.strip()]
        self.word_counts = Counter()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
    def animate_word_frequency(self, frame):
        """Animate word frequency building up sentence by sentence"""
        if frame < len(self.sentences):
            # Process current sentence
            current_sentence = self.sentences[frame]
            words = current_sentence.lower().split()
            
            # Update word counts
            for word in words:
                # Remove punctuation
                clean_word = ''.join(char for char in word if char.isalnum())
                if clean_word and len(clean_word) > 2:  # Skip short words
                    self.word_counts[clean_word] += 1
            
            # Clear axes
            self.ax1.clear()
            self.ax2.clear()
            
            # Plot 1: Current sentence processing
            self.ax1.text(0.5, 0.7, f"Processing Sentence {frame + 1}:", 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=12, fontweight='bold')
            self.ax1.text(0.5, 0.5, f'"{current_sentence}"', 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=10, style='italic', wrap=True)
            self.ax1.text(0.5, 0.3, f"New words found: {len(words)}", 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=10)
            self.ax1.set_title("Current Processing", fontweight='bold')
            self.ax1.axis('off')
            
            # Plot 2: Word frequency bar chart
            if self.word_counts:
                # Get top 10 most frequent words
                top_words = self.word_counts.most_common(10)
                words, counts = zip(*top_words)
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
                bars = self.ax2.bar(range(len(words)), counts, color=colors)
                
                self.ax2.set_xlabel('Words')
                self.ax2.set_ylabel('Frequency')
                self.ax2.set_title(f'Top Word Frequencies (Total: {sum(self.word_counts.values())} words)')
                self.ax2.set_xticks(range(len(words)))
                self.ax2.set_xticklabels(words, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    self.ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{count}', ha='center', va='bottom')
            
            plt.tight_layout()
        
        return []

def demonstrate_word_frequency():
    """Run the word frequency animation demo"""
    sample_text = """
    Natural language processing is a fascinating field. 
    Machine learning algorithms help computers understand human language.
    Text analysis involves many techniques like tokenization and parsing.
    Python provides excellent libraries for natural language processing.
    Data scientists use NLP to extract insights from text data.
    Understanding language patterns helps improve machine learning models.
    Text preprocessing is crucial for effective natural language processing.
    """
    
    print("🎬 Starting Word Frequency Animation Demo!")
    print("=" * 50)
    
    animator = WordFrequencyAnimator(sample_text)
    
    # Create animation
    anim = animation.FuncAnimation(
        animator.fig, 
        animator.animate_word_frequency,
        frames=len(animator.sentences) + 2,
        interval=2000,  # 2 seconds per frame
        repeat=True,
        blit=False
    )
    
    plt.show()
    
    # Print final statistics
    print("\nFinal Word Frequency Analysis:")
    print("-" * 30)
    for word, count in animator.word_counts.most_common(10):
        print(f"{word:15} : {count:2d}")

if __name__ == "__main__":
    demonstrate_word_frequency()
