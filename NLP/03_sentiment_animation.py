"""
Animated Sentiment Analysis Demo
This script visualizes how sentiment changes across different texts
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from textblob import TextBlob
import time

class SentimentAnimator:
    def __init__(self, texts):
        self.texts = texts
        self.sentiments = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            return polarity, subjectivity
        except:
            # Fallback simple sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'disappointed', 'horrible', 'worst']
            
            words = text.lower().split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            if pos_count + neg_count == 0:
                return 0, 0.5
            
            polarity = (pos_count - neg_count) / len(words)
            subjectivity = (pos_count + neg_count) / len(words)
            
            return polarity, min(subjectivity, 1.0)
    
    def get_sentiment_label(self, polarity):
        """Convert polarity to readable label"""
        if polarity > 0.1:
            return "Positive 😊", "green"
        elif polarity < -0.1:
            return "Negative 😞", "red"
        else:
            return "Neutral 😐", "gray"
    
    def animate_sentiment(self, frame):
        """Animate sentiment analysis for each text"""
        if frame < len(self.texts):
            current_text = self.texts[frame]
            polarity, subjectivity = self.analyze_sentiment(current_text)
            self.sentiments.append((polarity, subjectivity))
            
            # Clear axes
            self.ax1.clear()
            self.ax2.clear()
            
            # Plot 1: Current text analysis
            sentiment_label, color = self.get_sentiment_label(polarity)
            
            self.ax1.text(0.5, 0.8, f"Text {frame + 1}:", 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=14, fontweight='bold')
            
            # Wrap long text
            wrapped_text = current_text[:100] + "..." if len(current_text) > 100 else current_text
            self.ax1.text(0.5, 0.6, f'"{wrapped_text}"', 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=10, style='italic')
            
            self.ax1.text(0.5, 0.4, f"Sentiment: {sentiment_label}", 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=12, fontweight='bold', color=color)
            
            self.ax1.text(0.5, 0.2, f"Polarity: {polarity:.2f} | Subjectivity: {subjectivity:.2f}", 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=10)
            
            self.ax1.set_title("Current Text Analysis", fontweight='bold')
            self.ax1.axis('off')
            
            # Plot 2: Sentiment progression
            if self.sentiments:
                polarities = [s[0] for s in self.sentiments]
                subjectivities = [s[1] for s in self.sentiments]
                
                x_range = range(1, len(polarities) + 1)
                
                # Plot polarity line
                line1 = self.ax2.plot(x_range, polarities, 'o-', color='blue', 
                                     linewidth=2, markersize=8, label='Polarity')
                
                # Color code the points
                for i, (x, pol) in enumerate(zip(x_range, polarities)):
                    _, point_color = self.get_sentiment_label(pol)
                    self.ax2.plot(x, pol, 'o', color=point_color, markersize=10, alpha=0.7)
                
                # Add horizontal reference lines
                self.ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Neutral')
                self.ax2.axhline(y=0.1, color='green', linestyle=':', alpha=0.5, label='Positive threshold')
                self.ax2.axhline(y=-0.1, color='red', linestyle=':', alpha=0.5, label='Negative threshold')
                
                self.ax2.set_xlabel('Text Number')
                self.ax2.set_ylabel('Sentiment Polarity')
                self.ax2.set_title('Sentiment Analysis Progression')
                self.ax2.set_ylim(-1.1, 1.1)
                self.ax2.grid(True, alpha=0.3)
                self.ax2.legend()
                
                # Add value labels
                for x, pol in zip(x_range, polarities):
                    self.ax2.annotate(f'{pol:.2f}', (x, pol), 
                                    textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
        
        return []

def demonstrate_sentiment_analysis():
    """Run the sentiment analysis animation demo"""
    sample_texts = [
        "I absolutely love this new Python library! It's amazing and so easy to use.",
        "This code is terrible and doesn't work at all. Very disappointing.",
        "The weather today is okay, nothing special but not bad either.",
        "Machine learning is fascinating and opens up incredible possibilities!",
        "I hate debugging errors, it's so frustrating and time-consuming.",
        "Natural language processing helps computers understand human communication.",
        "This tutorial is excellent and very well explained. Highly recommended!",
        "The performance is poor and the interface is confusing and unintuitive."
    ]
    
    print("🎭 Starting Sentiment Analysis Animation Demo!")
    print("=" * 50)
    
    animator = SentimentAnimator(sample_texts)
    
    # Create animation
    anim = animation.FuncAnimation(
        animator.fig, 
        animator.animate_sentiment,
        frames=len(sample_texts) + 1,
        interval=3000,  # 3 seconds per frame
        repeat=True,
        blit=False
    )
    
    plt.show()
    
    # Print final analysis
    print("\nFinal Sentiment Analysis Summary:")
    print("-" * 40)
    for i, (text, (pol, subj)) in enumerate(zip(sample_texts, animator.sentiments)):
        label, _ = animator.get_sentiment_label(pol)
        print(f"Text {i+1}: {label} (Polarity: {pol:.2f})")

if __name__ == "__main__":
    demonstrate_sentiment_analysis()
