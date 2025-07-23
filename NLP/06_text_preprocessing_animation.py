"""
Comprehensive Text Preprocessing Animation
This script demonstrates all major text preprocessing steps with detailed animations
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
import re
import string
from collections import Counter

class TextPreprocessingAnimator:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.preprocessing_steps = []
        self.step_names = [
            "Raw Text",
            "Lowercase Conversion", 
            "Remove Special Characters",
            "Remove Extra Whitespace",
            "Tokenization",
            "Remove Stop Words",
            "Remove Short Words",
            "Stemming/Lemmatization",
            "Final Clean Text"
        ]
        
        # Common stop words (simplified list)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Simple stemming rules (basic Porter stemmer patterns)
        self.stemming_rules = {
            'running': 'run', 'runs': 'run', 'ran': 'run',
            'walking': 'walk', 'walks': 'walk', 'walked': 'walk',
            'eating': 'eat', 'eats': 'eat', 'eaten': 'eat',
            'playing': 'play', 'plays': 'play', 'played': 'play',
            'working': 'work', 'works': 'work', 'worked': 'work',
            'learning': 'learn', 'learns': 'learn', 'learned': 'learn',
            'processing': 'process', 'processes': 'process', 'processed': 'process'
        }
        
        self.fig, self.axes = plt.subplots(3, 1, figsize=(16, 12))
        self.ax_input = self.axes[0]
        self.ax_process = self.axes[1] 
        self.ax_output = self.axes[2]
        
        # Perform all preprocessing steps
        self.perform_preprocessing()
        
    def perform_preprocessing(self):
        """Perform all preprocessing steps and store intermediate results"""
        current_text = self.raw_text
        self.preprocessing_steps.append(("Raw Text", current_text, "Original input text"))
        
        # Step 1: Lowercase conversion
        current_text = current_text.lower()
        self.preprocessing_steps.append(("Lowercase", current_text, "Convert all characters to lowercase"))
        
        # Step 2: Remove special characters (keep letters, numbers, spaces)
        current_text = re.sub(r'[^a-zA-Z0-9\s]', '', current_text)
        self.preprocessing_steps.append(("Remove Special Chars", current_text, "Remove punctuation and special characters"))
        
        # Step 3: Remove extra whitespace
        current_text = re.sub(r'\s+', ' ', current_text).strip()
        self.preprocessing_steps.append(("Clean Whitespace", current_text, "Remove extra spaces and normalize whitespace"))
        
        # Step 4: Tokenization
        tokens = current_text.split()
        self.preprocessing_steps.append(("Tokenization", tokens, "Split text into individual words"))
        
        # Step 5: Remove stop words
        tokens_no_stop = [token for token in tokens if token.lower() not in self.stop_words]
        self.preprocessing_steps.append(("Remove Stop Words", tokens_no_stop, "Remove common words with little meaning"))
        
        # Step 6: Remove short words (length < 3)
        tokens_no_short = [token for token in tokens_no_stop if len(token) >= 3]
        self.preprocessing_steps.append(("Remove Short Words", tokens_no_short, "Remove words shorter than 3 characters"))
        
        # Step 7: Stemming/Lemmatization
        stemmed_tokens = [self.stemming_rules.get(token, token) for token in tokens_no_short]
        self.preprocessing_steps.append(("Stemming", stemmed_tokens, "Reduce words to their root form"))
        
        # Step 8: Final result
        final_text = ' '.join(stemmed_tokens)
        self.preprocessing_steps.append(("Final Result", final_text, "Clean, processed text ready for analysis"))
    
    def animate_preprocessing(self, frame):
        """Animate each preprocessing step"""
        # Clear all axes
        for ax in self.axes:
            ax.clear()
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
        
        if frame < len(self.preprocessing_steps):
            step_name, step_data, step_description = self.preprocessing_steps[frame]
            
            # Show current step information
            self.show_step_header(frame, step_name, step_description)
            
            # Show input (previous step or original)
            if frame > 0:
                prev_step_name, prev_data, _ = self.preprocessing_steps[frame-1]
                self.show_input_text(prev_step_name, prev_data)
            else:
                self.show_input_text("Input", self.raw_text)
            
            # Show processing visualization
            self.show_processing_step(frame, step_name, step_data)
            
            # Show output
            self.show_output_text(step_name, step_data)
            
            # Show statistics
            self.show_statistics(frame)
        
        return []
    
    def show_step_header(self, step_num, step_name, description):
        """Show the current step information"""
        title = f"Text Preprocessing Step {step_num + 1}/{len(self.preprocessing_steps)}: {step_name}"
        self.fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Add description
        self.ax_input.text(5, 9, description, ha='center', va='center', 
                          fontsize=12, style='italic',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    
    def show_input_text(self, label, text_data):
        """Show input text or tokens"""
        self.ax_input.text(1, 7, f"{label}:", fontsize=12, fontweight='bold')
        
        if isinstance(text_data, str):
            # Display text with character highlighting
            self.display_text_with_highlighting(self.ax_input, text_data, y_pos=5, 
                                              color='lightblue', label="Input Text")
        else:
            # Display tokens
            self.display_tokens(self.ax_input, text_data, y_pos=5, 
                              color='lightblue', label="Input Tokens")
    
    def show_processing_step(self, step_num, step_name, step_data):
        """Show the processing visualization"""
        if step_name == "Lowercase":
            self.show_lowercase_process()
        elif step_name == "Remove Special Chars":
            self.show_special_char_removal()
        elif step_name == "Clean Whitespace":
            self.show_whitespace_cleaning()
        elif step_name == "Tokenization":
            self.show_tokenization_process()
        elif step_name == "Remove Stop Words":
            self.show_stop_word_removal(step_data)
        elif step_name == "Remove Short Words":
            self.show_short_word_removal(step_data)
        elif step_name == "Stemming":
            self.show_stemming_process(step_data)
        else:
            self.show_generic_process(step_name)
    
    def show_lowercase_process(self):
        """Visualize lowercase conversion"""
        self.ax_process.text(5, 8, "Converting to Lowercase", ha='center', fontsize=14, fontweight='bold')
        
        # Show transformation examples
        examples = [("Hello", "hello"), ("WORLD", "world"), ("NLP", "nlp")]
        x_positions = [2, 5, 8]
        
        for (original, converted), x_pos in zip(examples, x_positions):
            # Original
            self.ax_process.text(x_pos, 6, original, ha='center', va='center', fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            
            # Arrow
            self.ax_process.annotate('', xy=(x_pos, 4), xytext=(x_pos, 5.5),
                                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
            
            # Converted
            self.ax_process.text(x_pos, 3, converted, ha='center', va='center', fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    def show_special_char_removal(self):
        """Visualize special character removal"""
        self.ax_process.text(5, 8, "Removing Special Characters", ha='center', fontsize=14, fontweight='bold')
        
        # Show what gets removed vs kept
        self.ax_process.text(2.5, 6, "REMOVE", ha='center', fontsize=12, fontweight='bold', color='red')
        self.ax_process.text(7.5, 6, "KEEP", ha='center', fontsize=12, fontweight='bold', color='green')
        
        remove_chars = ['!', '@', '#', '$', '%', '&', '*', '(', ')', '-', '+', '=']
        keep_chars = ['a', 'b', 'c', '1', '2', '3', ' ']
        
        # Show characters to remove
        for i, char in enumerate(remove_chars[:6]):
            x_pos = 1 + (i % 3) * 0.5
            y_pos = 4.5 - (i // 3) * 0.5
            self.ax_process.text(x_pos, y_pos, char, ha='center', va='center', fontsize=10,
                               bbox=dict(boxstyle="circle,pad=0.1", facecolor="lightcoral"))
        
        # Show characters to keep
        for i, char in enumerate(keep_chars):
            x_pos = 6.5 + (i % 4) * 0.5
            y_pos = 4.5 - (i // 4) * 0.5
            self.ax_process.text(x_pos, y_pos, char, ha='center', va='center', fontsize=10,
                               bbox=dict(boxstyle="circle,pad=0.1", facecolor="lightgreen"))
    
    def show_whitespace_cleaning(self):
        """Visualize whitespace cleaning"""
        self.ax_process.text(5, 8, "Cleaning Whitespace", ha='center', fontsize=14, fontweight='bold')
        
        # Show before and after
        before = "hello    world  \n  nlp   "
        after = "hello world nlp"
        
        self.ax_process.text(5, 6, "Before:", ha='center', fontsize=12, fontweight='bold')
        self.ax_process.text(5, 5.5, f'"{before}"', ha='center', fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        self.ax_process.annotate('', xy=(5, 4), xytext=(5, 5),
                               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        self.ax_process.text(5, 3.5, "After:", ha='center', fontsize=12, fontweight='bold')
        self.ax_process.text(5, 3, f'"{after}"', ha='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    def show_tokenization_process(self):
        """Visualize tokenization"""
        self.ax_process.text(5, 8, "Tokenization Process", ha='center', fontsize=14, fontweight='bold')
        
        # Show text being split
        sample_text = "natural language processing"
        tokens = sample_text.split()
        
        # Show original text
        self.ax_process.text(5, 6.5, f'"{sample_text}"', ha='center', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Show splitting arrows
        self.ax_process.text(5, 5.5, "↓ SPLIT ↓", ha='center', fontsize=10, fontweight='bold')
        
        # Show individual tokens
        x_positions = np.linspace(2, 8, len(tokens))
        for token, x_pos in zip(tokens, x_positions):
            self.ax_process.text(x_pos, 4, f'"{token}"', ha='center', va='center', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
        
        self.ax_process.text(5, 2.5, f"Result: {len(tokens)} tokens", ha='center', fontsize=12, fontweight='bold')
    
    def show_stop_word_removal(self, remaining_tokens):
        """Visualize stop word removal"""
        self.ax_process.text(5, 8, "Removing Stop Words", ha='center', fontsize=14, fontweight='bold')
        
        # Get previous tokens
        if len(self.preprocessing_steps) > 5:
            prev_tokens = self.preprocessing_steps[4][1]  # Tokenization step
            
            # Show which words are removed vs kept
            x_positions = np.linspace(1, 9, len(prev_tokens))
            
            for token, x_pos in zip(prev_tokens, x_positions):
                if token.lower() in self.stop_words:
                    # Stop word - will be removed
                    color = "lightcoral"
                    label = "REMOVE"
                    text_color = "red"
                else:
                    # Content word - will be kept
                    color = "lightgreen" 
                    label = "KEEP"
                    text_color = "green"
                
                self.ax_process.text(x_pos, 5, token, ha='center', va='center', fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color))
                self.ax_process.text(x_pos, 4, label, ha='center', va='center', fontsize=8,
                                   color=text_color, fontweight='bold')
            
            removed_count = len(prev_tokens) - len(remaining_tokens)
            self.ax_process.text(5, 2, f"Removed {removed_count} stop words, kept {len(remaining_tokens)} content words", 
                               ha='center', fontsize=12, fontweight='bold')
    
    def show_short_word_removal(self, remaining_tokens):
        """Visualize short word removal"""
        self.ax_process.text(5, 8, "Removing Short Words (< 3 characters)", ha='center', fontsize=14, fontweight='bold')
        
        # Get previous tokens
        if len(self.preprocessing_steps) > 6:
            prev_tokens = self.preprocessing_steps[5][1]  # After stop word removal
            
            x_positions = np.linspace(1, 9, len(prev_tokens))
            
            for token, x_pos in zip(prev_tokens, x_positions):
                if len(token) < 3:
                    color = "lightcoral"
                    label = f"REMOVE\n(len={len(token)})"
                    text_color = "red"
                else:
                    color = "lightgreen"
                    label = f"KEEP\n(len={len(token)})"
                    text_color = "green"
                
                self.ax_process.text(x_pos, 5, token, ha='center', va='center', fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color))
                self.ax_process.text(x_pos, 3.5, label, ha='center', va='center', fontsize=8,
                                   color=text_color, fontweight='bold')
            
            removed_count = len(prev_tokens) - len(remaining_tokens)
            self.ax_process.text(5, 2, f"Removed {removed_count} short words, kept {len(remaining_tokens)} words", 
                               ha='center', fontsize=12, fontweight='bold')
    
    def show_stemming_process(self, stemmed_tokens):
        """Visualize stemming process"""
        self.ax_process.text(5, 8, "Stemming/Lemmatization", ha='center', fontsize=14, fontweight='bold')
        
        # Get previous tokens
        if len(self.preprocessing_steps) > 7:
            prev_tokens = self.preprocessing_steps[6][1]  # After short word removal
            
            # Show transformations
            transformations = []
            for i, (original, stemmed) in enumerate(zip(prev_tokens, stemmed_tokens)):
                if original != stemmed:
                    transformations.append((original, stemmed))
            
            if transformations:
                self.ax_process.text(5, 6.5, "Word Transformations:", ha='center', fontsize=12, fontweight='bold')
                
                # Show up to 4 transformations
                for i, (original, stemmed) in enumerate(transformations[:4]):
                    x_pos = 2 + i * 2
                    
                    # Original word
                    self.ax_process.text(x_pos, 5, original, ha='center', va='center', fontsize=10,
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue"))
                    
                    # Arrow
                    self.ax_process.annotate('', xy=(x_pos, 3.5), xytext=(x_pos, 4.5),
                                           arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
                    
                    # Stemmed word
                    self.ax_process.text(x_pos, 3, stemmed, ha='center', va='center', fontsize=10,
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
                
                changed_count = len(transformations)
                self.ax_process.text(5, 1.5, f"Stemmed {changed_count} words to their root forms", 
                                   ha='center', fontsize=12, fontweight='bold')
            else:
                self.ax_process.text(5, 4, "No stemming transformations needed", 
                                   ha='center', fontsize=12, style='italic')
    
    def show_generic_process(self, step_name):
        """Show generic processing step"""
        self.ax_process.text(5, 5, f"Processing: {step_name}", ha='center', fontsize=14, fontweight='bold')
    
    def show_output_text(self, step_name, step_data):
        """Show output text or tokens"""
        self.ax_output.text(1, 8, f"Output ({step_name}):", fontsize=12, fontweight='bold')
        
        if isinstance(step_data, str):
            self.display_text_with_highlighting(self.ax_output, step_data, y_pos=6, 
                                              color='lightgreen', label="Output Text")
        else:
            self.display_tokens(self.ax_output, step_data, y_pos=6, 
                              color='lightgreen', label="Output Tokens")
    
    def display_text_with_highlighting(self, ax, text, y_pos, color, label):
        """Display text with character-level highlighting"""
        if len(text) > 80:
            display_text = text[:77] + "..."
        else:
            display_text = text
            
        ax.text(5, y_pos, f'"{display_text}"', ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.5", facecolor=color),
               wrap=True)
        
        # Show character count
        ax.text(5, y_pos-1.5, f"Length: {len(text)} characters", 
               ha='center', va='center', fontsize=10, style='italic')
    
    def display_tokens(self, ax, tokens, y_pos, color, label):
        """Display tokens as individual boxes"""
        if not tokens:
            ax.text(5, y_pos, "No tokens", ha='center', va='center', fontsize=12, style='italic')
            return
            
        # Show up to 8 tokens, then indicate more
        display_tokens = tokens[:8]
        
        if len(tokens) <= 8:
            x_positions = np.linspace(1, 9, len(display_tokens))
        else:
            x_positions = np.linspace(1, 8, len(display_tokens))
        
        for token, x_pos in zip(display_tokens, x_positions):
            ax.text(x_pos, y_pos, token, ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color))
        
        if len(tokens) > 8:
            ax.text(9, y_pos, f"... +{len(tokens)-8} more", ha='center', va='center', 
                   fontsize=10, style='italic')
        
        # Show token count
        ax.text(5, y_pos-1.5, f"Token count: {len(tokens)}", 
               ha='center', va='center', fontsize=10, style='italic')
    
    def show_statistics(self, step_num):
        """Show preprocessing statistics"""
        if step_num == 0:
            return
            
        # Calculate statistics
        original_length = len(self.raw_text)
        current_step_name, current_data, _ = self.preprocessing_steps[step_num]
        
        if isinstance(current_data, str):
            current_length = len(current_data)
            reduction = ((original_length - current_length) / original_length) * 100
            stats_text = f"Size reduction: {reduction:.1f}% ({original_length} → {current_length} chars)"
        else:
            current_length = len(current_data)
            original_tokens = len(self.raw_text.split())
            if original_tokens > 0:
                reduction = ((original_tokens - current_length) / original_tokens) * 100
                stats_text = f"Token reduction: {reduction:.1f}% ({original_tokens} → {current_length} tokens)"
            else:
                stats_text = f"Tokens: {current_length}"
        
        # Show progress bar
        progress = (step_num + 1) / len(self.preprocessing_steps)
        self.show_progress_bar(progress, stats_text)
    
    def show_progress_bar(self, progress, stats_text):
        """Show preprocessing progress"""
        # Progress bar background
        bar_width = 6
        bar_height = 0.3
        bar_x = 2
        bar_y = 0.5
        
        bg_rect = Rectangle((bar_x, bar_y), bar_width, bar_height, 
                           facecolor='lightgray', edgecolor='black')
        self.ax_output.add_patch(bg_rect)
        
        # Progress fill
        fill_width = bar_width * progress
        fill_rect = Rectangle((bar_x, bar_y), fill_width, bar_height,
                             facecolor='green', alpha=0.7)
        self.ax_output.add_patch(fill_rect)
        
        # Progress text
        self.ax_output.text(5, 0.65, f"Progress: {progress*100:.0f}%", 
                          ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Statistics
        self.ax_output.text(5, 0.2, stats_text, ha='center', va='center', 
                          fontsize=9, style='italic')

def demonstrate_text_preprocessing():
    """Run the text preprocessing animation demo"""
    sample_texts = [
        "Hello World! This is a SAMPLE text with Numbers123, punctuation!!! and    extra spaces.",
        "Natural Language Processing (NLP) is AMAZING!!! It's used for analyzing text data & extracting insights.",
        "Machine Learning algorithms are POWERFUL tools!!! They can process & analyze large datasets efficiently."
    ]
    
    print("🔧 Starting Text Preprocessing Animation Demo!")
    print("=" * 70)
    
    for i, text in enumerate(sample_texts):
        print(f"\nDemo {i+1}: Processing text...")
        print(f"Original: '{text}'")
        
        animator = TextPreprocessingAnimator(text)
        
        # Create animation
        anim = animation.FuncAnimation(
            animator.fig,
            animator.animate_preprocessing,
            frames=len(animator.preprocessing_steps),
            interval=4000,  # 4 seconds per frame
            repeat=True,
            blit=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # Print final comparison
        print("\nPreprocessing Results:")
        print(f"Original:  '{text}'")
        final_result = animator.preprocessing_steps[-1][1]
        print(f"Processed: '{final_result}'")
        print(f"Reduction: {len(text)} → {len(final_result)} characters")
        print("-" * 70)

def explain_preprocessing_concepts():
    """Explain text preprocessing concepts"""
    print("\n" + "="*70)
    print("🔧 TEXT PREPROCESSING CONCEPTS")
    print("="*70)
    print()
    print("🎯 WHY PREPROCESS TEXT?")
    print("   - Raw text is messy and inconsistent")
    print("   - Reduces noise and focuses on meaningful content")
    print("   - Standardizes format for machine learning")
    print("   - Improves model performance and accuracy")
    print()
    print("📋 COMMON PREPROCESSING STEPS:")
    print()
    print("1. 🔤 LOWERCASE CONVERSION")
    print("   • Standardizes text case")
    print("   • 'Hello' and 'hello' become the same")
    print("   • Reduces vocabulary size")
    print()
    print("2. 🧹 REMOVE SPECIAL CHARACTERS")
    print("   • Eliminates punctuation and symbols")
    print("   • Keeps only letters, numbers, spaces")
    print("   • Reduces noise in text")
    print()
    print("3. ⚡ CLEAN WHITESPACE")
    print("   • Removes extra spaces and tabs")
    print("   • Normalizes spacing between words")
    print("   • Cleans up formatting issues")
    print()
    print("4. ✂️ TOKENIZATION")
    print("   • Splits text into individual words")
    print("   • Creates list of tokens for processing")
    print("   • Foundation for further analysis")
    print()
    print("5. 🚫 REMOVE STOP WORDS")
    print("   • Eliminates common words (the, and, is)")
    print("   • Focuses on content-bearing words")
    print("   • Reduces dimensionality")
    print()
    print("6. 📏 REMOVE SHORT WORDS")
    print("   • Filters out very short words")
    print("   • Removes abbreviations and noise")
    print("   • Keeps meaningful terms")
    print()
    print("7. 🌱 STEMMING/LEMMATIZATION")
    print("   • Reduces words to root forms")
    print("   • 'running', 'runs' → 'run'")
    print("   • Groups related word forms")
    print()
    print("💡 BEST PRACTICES:")
    print("   - Order matters! Follow logical sequence")
    print("   - Consider your specific use case")
    print("   - Balance cleaning vs. information loss")
    print("   - Test impact on model performance")
    print("   - Document your preprocessing pipeline")
    print("="*70)

if __name__ == "__main__":
    explain_preprocessing_concepts()
    demonstrate_text_preprocessing()
