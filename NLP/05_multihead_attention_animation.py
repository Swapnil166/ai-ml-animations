"""
Multi-Head Attention Animation
This script visualizes how multiple attention heads work together in transformers
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

class MultiHeadAttentionAnimator:
    def __init__(self, sentence="The quick brown fox jumps", num_heads=4):
        self.sentence = sentence
        self.tokens = sentence.split()
        self.n_tokens = len(self.tokens)
        self.num_heads = num_heads
        
        # Create different attention patterns for each head
        self.attention_heads = self.create_attention_heads()
        
        self.fig = plt.figure(figsize=(18, 12))
        self.setup_subplots()
        
        # Colors for different heads
        self.head_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    def create_attention_heads(self):
        """Create different attention patterns for each head"""
        heads = []
        
        for head_idx in range(self.num_heads):
            matrix = np.random.rand(self.n_tokens, self.n_tokens) * 0.2
            
            if head_idx == 0:  # Syntactic head - focuses on adjacent words
                for i in range(self.n_tokens):
                    matrix[i, i] = 0.6
                    if i > 0:
                        matrix[i, i-1] = 0.3
                    if i < self.n_tokens - 1:
                        matrix[i, i+1] = 0.3
                        
            elif head_idx == 1:  # Semantic head - focuses on content words
                content_positions = [0, 2, 3, 4]  # "The", "brown", "fox", "jumps"
                for i in content_positions:
                    for j in content_positions:
                        if i < self.n_tokens and j < self.n_tokens:
                            matrix[i, j] = 0.4 + np.random.rand() * 0.3
                            
            elif head_idx == 2:  # Positional head - focuses on position patterns
                for i in range(self.n_tokens):
                    matrix[i, i] = 0.8
                    # Attention to first and last tokens
                    matrix[i, 0] = 0.2
                    matrix[i, -1] = 0.2
                    
            else:  # Random/diverse head
                matrix = np.random.rand(self.n_tokens, self.n_tokens) * 0.5
                for i in range(self.n_tokens):
                    matrix[i, i] = 0.4 + np.random.rand() * 0.4
            
            # Normalize to sum to 1
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            heads.append(matrix)
        
        return heads
    
    def setup_subplots(self):
        """Setup subplot layout for multi-head visualization"""
        gs = self.fig.add_gridspec(3, self.num_heads, 
                                  height_ratios=[1, 2, 1], 
                                  hspace=0.3, wspace=0.2)
        
        self.ax_input = self.fig.add_subplot(gs[0, :])
        self.ax_heads = [self.fig.add_subplot(gs[1, i]) for i in range(self.num_heads)]
        self.ax_output = self.fig.add_subplot(gs[2, :])
    
    def animate_multihead_attention(self, frame):
        """Animate multi-head attention mechanism"""
        # Clear all axes
        self.ax_input.clear()
        self.ax_output.clear()
        for ax in self.ax_heads:
            ax.clear()
        
        if frame == 0:
            self.show_input_tokens()
            self.show_title("Multi-Head Attention: Input Tokens")
            
        elif frame == 1:
            self.show_input_tokens()
            self.show_head_explanation()
            self.show_title("Different Attention Heads Focus on Different Patterns")
            
        elif frame >= 2 and frame < 2 + self.num_heads:
            # Show each head individually
            head_idx = frame - 2
            self.show_input_tokens()
            self.show_single_head(head_idx)
            self.show_title(f"Head {head_idx + 1}: {self.get_head_description(head_idx)}")
            
        elif frame == 2 + self.num_heads:
            # Show all heads together
            self.show_input_tokens()
            self.show_all_heads()
            self.show_title("All Attention Heads Working Together")
            
        elif frame == 3 + self.num_heads:
            # Show concatenation and final output
            self.show_input_tokens()
            self.show_all_heads()
            self.show_concatenation_output()
            self.show_title("Concatenate Head Outputs → Final Representation")
        
        # Set up axes
        self.ax_input.set_xlim(0, 10)
        self.ax_input.set_ylim(0, 2)
        self.ax_input.axis('off')
        
        self.ax_output.set_xlim(0, 10)
        self.ax_output.set_ylim(0, 2)
        self.ax_output.axis('off')
        
        for ax in self.ax_heads:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
        
        return []
    
    def show_title(self, title):
        """Show the current step title"""
        self.fig.suptitle(f"🤖 Multi-Head Attention Mechanism\n{title}", 
                         fontsize=16, fontweight='bold', y=0.95)
    
    def show_input_tokens(self):
        """Display input tokens"""
        self.ax_input.text(5, 1.5, "Input Tokens:", ha='center', fontsize=12, fontweight='bold')
        
        x_positions = np.linspace(1, 9, self.n_tokens)
        
        for i, (token, x_pos) in enumerate(zip(self.tokens, x_positions)):
            box = FancyBboxPatch((x_pos-0.3, 0.7), 0.6, 0.4,
                               boxstyle="round,pad=0.05",
                               facecolor='lightblue', edgecolor='black', linewidth=2)
            self.ax_input.add_patch(box)
            
            self.ax_input.text(x_pos, 0.9, token, ha='center', va='center', 
                             fontweight='bold', fontsize=10)
            self.ax_input.text(x_pos, 0.5, f"pos {i}", ha='center', va='center', 
                             fontsize=8, style='italic')
    
    def get_head_description(self, head_idx):
        """Get description for each attention head"""
        descriptions = [
            "Syntactic Patterns (adjacent words)",
            "Semantic Relationships (content words)", 
            "Positional Information (position-based)",
            "Diverse Patterns (mixed focus)"
        ]
        return descriptions[head_idx] if head_idx < len(descriptions) else f"Head {head_idx + 1}"
    
    def show_head_explanation(self):
        """Show explanation of different head types"""
        explanations = [
            "Head 1: Syntactic - focuses on grammar and adjacent words",
            "Head 2: Semantic - captures meaning relationships",
            "Head 3: Positional - tracks position-based patterns", 
            "Head 4: Diverse - learns various other patterns"
        ]
        
        for i, (ax, explanation, color) in enumerate(zip(self.ax_heads, explanations, self.head_colors)):
            ax.text(5, 8, f"Head {i+1}", ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=color)
            ax.text(5, 6, explanation, ha='center', va='center', 
                   fontsize=10, wrap=True)
            
            # Draw a sample attention pattern
            self.draw_sample_pattern(ax, i, color)
    
    def draw_sample_pattern(self, ax, head_idx, color):
        """Draw a sample attention pattern for explanation"""
        x_positions = np.linspace(1, 9, self.n_tokens)
        
        # Draw tokens
        for i, (token, x_pos) in enumerate(zip(self.tokens, x_positions)):
            circle = plt.Circle((x_pos, 3), 0.3, facecolor='lightgray', 
                              edgecolor='black', alpha=0.7)
            ax.add_patch(circle)
            ax.text(x_pos, 3, token[:3], ha='center', va='center', fontsize=8)
        
        # Draw sample connections based on head type
        attention_matrix = self.attention_heads[head_idx]
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                weight = attention_matrix[i, j]
                if weight > 0.2:  # Only show significant connections
                    line_width = weight * 5
                    alpha = min(weight * 2, 0.8)
                    ax.plot([x_positions[i], x_positions[j]], [3, 3], 
                           linewidth=line_width, alpha=alpha, color=color)
    
    def show_single_head(self, head_idx):
        """Show detailed view of a single attention head"""
        attention_matrix = self.attention_heads[head_idx]
        color = self.head_colors[head_idx]
        
        # Show attention matrix as heatmap in the corresponding subplot
        ax = self.ax_heads[head_idx]
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='equal')
        
        ax.set_xticks(range(self.n_tokens))
        ax.set_yticks(range(self.n_tokens))
        ax.set_xticklabels(self.tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(self.tokens, fontsize=8)
        ax.set_title(f'Head {head_idx + 1}', fontweight='bold', color=color)
        
        # Add attention values
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                weight = attention_matrix[i, j]
                ax.text(j, i, f'{weight:.2f}', ha='center', va='center', 
                       fontsize=7, fontweight='bold',
                       color='white' if weight > 0.5 else 'black')
        
        # Clear other heads
        for other_idx, other_ax in enumerate(self.ax_heads):
            if other_idx != head_idx:
                other_ax.text(5, 5, f'Head {other_idx + 1}\n(waiting)', 
                            ha='center', va='center', fontsize=12, alpha=0.5)
    
    def show_all_heads(self):
        """Show all attention heads simultaneously"""
        for head_idx, (ax, attention_matrix, color) in enumerate(
            zip(self.ax_heads, self.attention_heads, self.head_colors)):
            
            # Show attention matrix
            im = ax.imshow(attention_matrix, cmap='Blues', aspect='equal', alpha=0.8)
            
            ax.set_xticks(range(self.n_tokens))
            ax.set_yticks(range(self.n_tokens))
            ax.set_xticklabels(self.tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(self.tokens, fontsize=8)
            ax.set_title(f'Head {head_idx + 1}', fontweight='bold', color=color)
            
            # Add border color to distinguish heads
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
    
    def show_concatenation_output(self):
        """Show the concatenation and final output"""
        self.ax_output.text(5, 1.5, "Final Output: Concatenated Multi-Head Attention", 
                          ha='center', fontsize=12, fontweight='bold')
        
        x_positions = np.linspace(1, 9, self.n_tokens)
        
        for i, (token, x_pos) in enumerate(zip(self.tokens, x_positions)):
            # Show enriched representation with multiple colors (representing different heads)
            box = FancyBboxPatch((x_pos-0.35, 0.6), 0.7, 0.5,
                               boxstyle="round,pad=0.05",
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
            self.ax_output.add_patch(box)
            
            # Add small colored strips to show multi-head contribution
            strip_width = 0.7 / self.num_heads
            for head_idx in range(self.num_heads):
                strip_x = x_pos - 0.35 + head_idx * strip_width
                strip = FancyBboxPatch((strip_x, 0.55), strip_width, 0.1,
                                     boxstyle="square,pad=0",
                                     facecolor=self.head_colors[head_idx], alpha=0.7)
                self.ax_output.add_patch(strip)
            
            self.ax_output.text(x_pos, 0.85, token, ha='center', va='center', 
                              fontweight='bold', fontsize=10)
            self.ax_output.text(x_pos, 0.4, "multi-head\nattended", ha='center', va='center', 
                              fontsize=7, style='italic')

def demonstrate_multihead_attention():
    """Run the multi-head attention animation demo"""
    sample_sentences = [
        "The quick brown fox jumps",
        "She reads interesting books daily",
        "AI models learn complex patterns"
    ]
    
    print("🤖 Starting Multi-Head Attention Mechanism Demo!")
    print("=" * 60)
    
    for i, sentence in enumerate(sample_sentences):
        print(f"\nDemo {i+1}: Analyzing '{sentence}'")
        
        animator = MultiHeadAttentionAnimator(sentence, num_heads=4)
        
        # Create animation
        total_frames = 4 + animator.num_heads + 1
        anim = animation.FuncAnimation(
            animator.fig,
            animator.animate_multihead_attention,
            frames=total_frames,
            interval=4000,  # 4 seconds per frame
            repeat=True,
            blit=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis for each head
        print("Multi-Head Attention Analysis:")
        for head_idx in range(animator.num_heads):
            print(f"\nHead {head_idx + 1} ({animator.get_head_description(head_idx)}):")
            attention_matrix = animator.attention_heads[head_idx]
            for j, token in enumerate(animator.tokens):
                top_attention = np.argsort(attention_matrix[j])[-2:][::-1]
                attention_info = [f"{animator.tokens[k]}({attention_matrix[j,k]:.2f})" 
                                for k in top_attention if k != j]
                if attention_info:
                    print(f"  {token} → {', '.join(attention_info)}")
        print("-" * 50)

def explain_multihead_concepts():
    """Explain multi-head attention concepts"""
    print("\n" + "="*60)
    print("🧠 MULTI-HEAD ATTENTION CONCEPTS")
    print("="*60)
    print()
    print("🎯 WHY MULTIPLE HEADS?")
    print("   - Different heads can focus on different types of relationships")
    print("   - Syntactic patterns (grammar, word order)")
    print("   - Semantic patterns (meaning, context)")
    print("   - Positional patterns (distance, structure)")
    print("   - Diverse patterns (learned automatically)")
    print()
    print("⚙️ HOW IT WORKS:")
    print("   1. Split Q, K, V into multiple 'heads'")
    print("   2. Each head computes attention independently")
    print("   3. Different heads learn different attention patterns")
    print("   4. Concatenate all head outputs")
    print("   5. Apply final linear transformation")
    print()
    print("🔍 HEAD SPECIALIZATION:")
    print("   • Head 1: Often focuses on syntax and adjacent words")
    print("   • Head 2: May capture semantic relationships")
    print("   • Head 3: Could track positional information")
    print("   • Head N: Learns other useful patterns")
    print()
    print("💪 ADVANTAGES:")
    print("   - Richer representation learning")
    print("   - Parallel processing of different patterns")
    print("   - Better handling of complex relationships")
    print("   - Improved model capacity without much overhead")
    print("="*60)

if __name__ == "__main__":
    explain_multihead_concepts()
    demonstrate_multihead_attention()
