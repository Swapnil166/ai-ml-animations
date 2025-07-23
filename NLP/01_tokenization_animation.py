"""
Animated Tokenization Demo
This script demonstrates how text tokenization works with step-by-step animation
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle
import time

class TokenizationAnimator:
    def __init__(self, text):
        self.text = text
        self.tokens = text.split()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.colors = plt.cm.Set3(np.linspace(0, 1, len(self.tokens)))
        
    def animate_tokenization(self):
        """Animate the tokenization process"""
        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 6)
        self.ax.set_title("NLP Tokenization Animation", fontsize=16, fontweight='bold')
        
        # Display original text
        self.ax.text(5, 5, f"Original Text: '{self.text}'", 
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Animate token separation
        y_pos = 3.5
        x_positions = np.linspace(1, 9, len(self.tokens))
        
        for i, (token, x_pos, color) in enumerate(zip(self.tokens, x_positions, self.colors)):
            # Add token box with animation effect
            rect = Rectangle((x_pos-0.4, y_pos-0.3), 0.8, 0.6, 
                           facecolor=color, alpha=0.7, edgecolor='black')
            self.ax.add_patch(rect)
            
            # Add token text
            self.ax.text(x_pos, y_pos, token, ha='center', va='center', 
                        fontweight='bold', fontsize=10)
            
            # Add token number
            self.ax.text(x_pos, y_pos-0.8, f"Token {i+1}", 
                        ha='center', va='center', fontsize=8, style='italic')
        
        # Add summary
        self.ax.text(5, 1.5, f"Total Tokens: {len(self.tokens)}", 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        self.ax.axis('off')
        plt.tight_layout()
        return self.ax.patches + self.ax.texts

def demonstrate_tokenization():
    """Run the tokenization animation demo"""
    sample_texts = [
        "Natural Language Processing is amazing!",
        "Machine learning helps computers understand text",
        "Python makes NLP accessible to everyone"
    ]
    
    for i, text in enumerate(sample_texts):
        print(f"\nDemo {i+1}: Tokenizing '{text}'")
        animator = TokenizationAnimator(text)
        animator.animate_tokenization()
        plt.show()
        
        # Print token analysis
        tokens = text.split()
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Average token length: {np.mean([len(token) for token in tokens]):.1f} characters")
        print("-" * 50)

if __name__ == "__main__":
    print("🚀 Starting NLP Tokenization Animation Demo!")
    print("=" * 50)
    demonstrate_tokenization()
