"""
Enhanced Tokenization Demo with Interactive Controls
This script demonstrates tokenization with user input and real-time processing
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.widgets import TextBox, Button
import tkinter as tk
from tkinter import simpledialog
import re

class InteractiveTokenizationDemo:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 10))
        self.fig.suptitle("🔤 Interactive Tokenization Demo", fontsize=16, fontweight='bold')
        
        # Flatten axes for easier access
        self.ax_input = self.axes[0, 0]
        self.ax_tokens = self.axes[0, 1] 
        self.ax_stats = self.axes[1, 0]
        self.ax_controls = self.axes[1, 1]
        
        # Default text
        self.current_text = "Natural Language Processing is amazing and powerful!"
        self.tokens = []
        self.token_stats = {}
        
        # Setup interactive elements
        self.setup_interface()
        self.process_text()
        
    def setup_interface(self):
        """Setup interactive controls"""
        # Clear controls axis and set up buttons
        self.ax_controls.clear()
        self.ax_controls.set_xlim(0, 10)
        self.ax_controls.set_ylim(0, 10)
        self.ax_controls.axis('off')
        
        # Add control buttons (simulated)
        self.ax_controls.text(5, 8, "Interactive Controls", ha='center', fontsize=14, fontweight='bold')
        self.ax_controls.text(5, 7, "Click 'New Text' to enter custom text", ha='center', fontsize=10)
        self.ax_controls.text(5, 6, "Press 'r' to reset to default", ha='center', fontsize=10)
        self.ax_controls.text(5, 5, "Press 's' to show statistics", ha='center', fontsize=10)
        
        # Add sample texts
        self.ax_controls.text(5, 3.5, "Sample Texts (click number):", ha='center', fontsize=12, fontweight='bold')
        samples = [
            "1. Hello world! How are you?",
            "2. Machine learning is fascinating.",
            "3. Python makes NLP accessible."
        ]
        
        for i, sample in enumerate(samples):
            self.ax_controls.text(5, 2.5-i*0.4, sample, ha='center', fontsize=9)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == 'r':
            self.current_text = "Natural Language Processing is amazing and powerful!"
            self.process_text()
        elif event.key == 's':
            self.show_detailed_stats()
        elif event.key == 'n':
            self.get_new_text()
        elif event.key in ['1', '2', '3']:
            samples = [
                "Hello world! How are you today?",
                "Machine learning algorithms are fascinating and powerful.",
                "Python makes natural language processing accessible to everyone."
            ]
            idx = int(event.key) - 1
            if idx < len(samples):
                self.current_text = samples[idx]
                self.process_text()
    
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes == self.ax_controls:
            # Simple click handling for sample texts
            if 1.5 <= event.ydata <= 2.5:
                if event.xdata < 5:
                    self.get_new_text()
    
    def get_new_text(self):
        """Get new text from user (simplified for demo)"""
        # In a real implementation, you'd use a proper input dialog
        print("Enter new text in the console:")
        try:
            new_text = input("Your text: ")
            if new_text.strip():
                self.current_text = new_text
                self.process_text()
        except:
            print("Using default text")
    
    def process_text(self):
        """Process the current text and update visualizations"""
        # Basic tokenization
        self.tokens = self.current_text.split()
        
        # Calculate statistics
        self.token_stats = {
            'total_tokens': len(self.tokens),
            'unique_tokens': len(set(self.tokens)),
            'avg_length': np.mean([len(token) for token in self.tokens]) if self.tokens else 0,
            'longest_token': max(self.tokens, key=len) if self.tokens else "",
            'shortest_token': min(self.tokens, key=len) if self.tokens else ""
        }
        
        # Update all visualizations
        self.update_input_display()
        self.update_token_display()
        self.update_stats_display()
        
        # Refresh the plot
        self.fig.canvas.draw()
    
    def update_input_display(self):
        """Update the input text display"""
        self.ax_input.clear()
        self.ax_input.set_xlim(0, 10)
        self.ax_input.set_ylim(0, 10)
        self.ax_input.axis('off')
        
        # Title
        self.ax_input.text(5, 9, "Input Text", ha='center', fontsize=14, fontweight='bold')
        
        # Text display with word wrapping
        wrapped_text = self.wrap_text(self.current_text, 50)
        y_pos = 7
        for line in wrapped_text:
            self.ax_input.text(5, y_pos, f'"{line}"', ha='center', va='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            y_pos -= 1
        
        # Character count
        self.ax_input.text(5, 2, f"Characters: {len(self.current_text)}", 
                          ha='center', fontsize=10, style='italic')
    
    def update_token_display(self):
        """Update the token visualization"""
        self.ax_tokens.clear()
        self.ax_tokens.set_xlim(0, 10)
        self.ax_tokens.set_ylim(0, 10)
        self.ax_tokens.axis('off')
        
        # Title
        self.ax_tokens.text(5, 9, "Tokens", ha='center', fontsize=14, fontweight='bold')
        
        if not self.tokens:
            self.ax_tokens.text(5, 5, "No tokens to display", ha='center', fontsize=12, style='italic')
            return
        
        # Display tokens in a grid
        max_tokens_per_row = 3
        rows = (len(self.tokens) + max_tokens_per_row - 1) // max_tokens_per_row
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tokens)))
        
        for i, (token, color) in enumerate(zip(self.tokens, colors)):
            row = i // max_tokens_per_row
            col = i % max_tokens_per_row
            
            # Calculate position
            x_pos = 2 + col * 2.5
            y_pos = 7.5 - row * 1.2
            
            if y_pos > 1:  # Only show if it fits
                # Token box
                self.ax_tokens.add_patch(
                    FancyBboxPatch((x_pos-0.4, y_pos-0.3), 0.8, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, alpha=0.7, edgecolor='black')
                )
                
                # Token text
                display_token = token[:8] + "..." if len(token) > 8 else token
                self.ax_tokens.text(x_pos, y_pos, display_token, ha='center', va='center',
                                  fontweight='bold', fontsize=9)
                
                # Token index
                self.ax_tokens.text(x_pos, y_pos-0.5, f"#{i+1}", ha='center', va='center',
                                  fontsize=7, style='italic')
        
        # Show total if there are more tokens
        if len(self.tokens) > 9:
            self.ax_tokens.text(5, 1, f"... and {len(self.tokens)-9} more tokens", 
                              ha='center', fontsize=10, style='italic')
    
    def update_stats_display(self):
        """Update the statistics display"""
        self.ax_stats.clear()
        self.ax_stats.set_xlim(0, 10)
        self.ax_stats.set_ylim(0, 10)
        self.ax_stats.axis('off')
        
        # Title
        self.ax_stats.text(5, 9, "Statistics", ha='center', fontsize=14, fontweight='bold')
        
        # Statistics
        stats_text = [
            f"Total tokens: {self.token_stats['total_tokens']}",
            f"Unique tokens: {self.token_stats['unique_tokens']}",
            f"Average length: {self.token_stats['avg_length']:.1f} chars",
            f"Longest token: '{self.token_stats['longest_token']}'",
            f"Shortest token: '{self.token_stats['shortest_token']}'"
        ]
        
        y_pos = 7.5
        for stat in stats_text:
            self.ax_stats.text(5, y_pos, stat, ha='center', va='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.5))
            y_pos -= 1
        
        # Token length distribution (simple bar chart)
        if self.tokens:
            lengths = [len(token) for token in self.tokens]
            unique_lengths = sorted(set(lengths))
            counts = [lengths.count(length) for length in unique_lengths]
            
            if len(unique_lengths) <= 5:  # Only show if not too many different lengths
                self.ax_stats.text(5, 2, "Token Length Distribution:", ha='center', fontsize=10, fontweight='bold')
                bar_width = 0.3
                x_positions = np.linspace(3, 7, len(unique_lengths))
                
                for x_pos, length, count in zip(x_positions, unique_lengths, counts):
                    # Simple bar representation
                    bar_height = count * 0.2
                    self.ax_stats.add_patch(
                        Rectangle((x_pos-bar_width/2, 0.5), bar_width, bar_height,
                                facecolor='orange', alpha=0.7, edgecolor='black')
                    )
                    self.ax_stats.text(x_pos, 0.3, str(length), ha='center', fontsize=8)
                    self.ax_stats.text(x_pos, 0.5 + bar_height + 0.1, str(count), ha='center', fontsize=8)
    
    def wrap_text(self, text, width):
        """Simple text wrapping"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def show_detailed_stats(self):
        """Show detailed statistics in console"""
        print("\n" + "="*50)
        print("DETAILED TOKENIZATION STATISTICS")
        print("="*50)
        print(f"Original text: '{self.current_text}'")
        print(f"Total characters: {len(self.current_text)}")
        print(f"Total tokens: {self.token_stats['total_tokens']}")
        print(f"Unique tokens: {self.token_stats['unique_tokens']}")
        print(f"Vocabulary diversity: {self.token_stats['unique_tokens']/self.token_stats['total_tokens']:.2f}")
        print(f"Average token length: {self.token_stats['avg_length']:.2f} characters")
        print(f"Longest token: '{self.token_stats['longest_token']}' ({len(self.token_stats['longest_token'])} chars)")
        print(f"Shortest token: '{self.token_stats['shortest_token']}' ({len(self.token_stats['shortest_token'])} chars)")
        print("\nAll tokens:")
        for i, token in enumerate(self.tokens, 1):
            print(f"  {i:2d}. '{token}' ({len(token)} chars)")
        print("="*50)

def run_interactive_demo():
    """Run the interactive tokenization demo"""
    print("🔤 Interactive Tokenization Demo")
    print("="*50)
    print("Controls:")
    print("  'r' - Reset to default text")
    print("  's' - Show detailed statistics")
    print("  'n' - Enter new text (console input)")
    print("  '1', '2', '3' - Load sample texts")
    print("  Click and explore the interface!")
    print("="*50)
    
    demo = InteractiveTokenizationDemo()
    plt.tight_layout()
    plt.show()
    
    return demo

if __name__ == "__main__":
    demo = run_interactive_demo()
