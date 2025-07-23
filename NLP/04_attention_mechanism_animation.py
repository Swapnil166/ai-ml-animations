"""
Animated Attention Mechanism in Transformers
This script visualizes how attention mechanisms work in transformer models
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.patches as mpatches

class AttentionAnimator:
    def __init__(self, sentence="The cat sat on the mat"):
        self.sentence = sentence
        self.tokens = sentence.split()
        self.n_tokens = len(self.tokens)
        
        # Create attention matrix (simplified for visualization)
        self.attention_matrix = self.create_attention_matrix()
        
        self.fig = plt.figure(figsize=(16, 10))
        self.setup_subplots()
        
    def create_attention_matrix(self):
        """Create a realistic attention matrix"""
        # Simulate attention patterns
        matrix = np.random.rand(self.n_tokens, self.n_tokens) * 0.3
        
        # Add some realistic patterns
        for i in range(self.n_tokens):
            # Self-attention (diagonal)
            matrix[i, i] = 0.8 + np.random.rand() * 0.2
            
            # Adjacent word attention
            if i > 0:
                matrix[i, i-1] = 0.4 + np.random.rand() * 0.3
            if i < self.n_tokens - 1:
                matrix[i, i+1] = 0.4 + np.random.rand() * 0.3
        
        # Normalize rows to sum to 1 (attention weights)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        
        return matrix
    
    def setup_subplots(self):
        """Setup the subplot layout"""
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], width_ratios=[2, 1])
        
        self.ax_input = self.fig.add_subplot(gs[0, :])      # Input tokens
        self.ax_attention = self.fig.add_subplot(gs[1, 0])   # Attention visualization
        self.ax_matrix = self.fig.add_subplot(gs[1, 1])      # Attention matrix
        self.ax_output = self.fig.add_subplot(gs[2, :])      # Output representation
        
    def animate_attention(self, frame):
        """Animate the attention mechanism step by step"""
        # Clear all axes
        for ax in [self.ax_input, self.ax_attention, self.ax_matrix, self.ax_output]:
            ax.clear()
        
        if frame == 0:
            self.show_input_tokens()
            self.show_title("Step 1: Input Tokens")
            
        elif frame == 1:
            self.show_input_tokens()
            self.show_qkv_computation()
            self.show_title("Step 2: Compute Query, Key, Value vectors")
            
        elif frame == 2:
            self.show_input_tokens()
            self.show_attention_computation()
            self.show_title("Step 3: Compute Attention Scores")
            
        elif frame >= 3 and frame < 3 + self.n_tokens:
            # Show attention for each token
            focus_token = frame - 3
            self.show_input_tokens(highlight=focus_token)
            self.show_attention_focus(focus_token)
            self.show_attention_matrix(highlight_row=focus_token)
            self.show_title(f"Step 4: Attention for '{self.tokens[focus_token]}'")
            
        elif frame == 3 + self.n_tokens:
            self.show_input_tokens()
            self.show_full_attention()
            self.show_attention_matrix()
            self.show_output_representation()
            self.show_title("Step 5: Final Attended Representation")
        
        # Remove axes for cleaner look
        for ax in [self.ax_input, self.ax_attention, self.ax_output]:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 2)
            ax.axis('off')
        
        self.ax_matrix.set_aspect('equal')
        
        return []
    
    def show_title(self, title):
        """Show the current step title"""
        self.fig.suptitle(f"🤖 Transformer Attention Mechanism\n{title}", 
                         fontsize=16, fontweight='bold', y=0.95)
    
    def show_input_tokens(self, highlight=None):
        """Display input tokens"""
        self.ax_input.text(5, 1.5, "Input Tokens:", ha='center', fontsize=12, fontweight='bold')
        
        x_positions = np.linspace(1, 9, self.n_tokens)
        colors = ['lightblue'] * self.n_tokens
        
        if highlight is not None:
            colors[highlight] = 'orange'
        
        for i, (token, x_pos, color) in enumerate(zip(self.tokens, x_positions, colors)):
            # Token box
            box = FancyBboxPatch((x_pos-0.3, 0.7), 0.6, 0.4,
                               boxstyle="round,pad=0.05", 
                               facecolor=color, edgecolor='black', linewidth=2)
            self.ax_input.add_patch(box)
            
            # Token text
            self.ax_input.text(x_pos, 0.9, token, ha='center', va='center', 
                             fontweight='bold', fontsize=10)
            
            # Position index
            self.ax_input.text(x_pos, 0.5, f"pos {i}", ha='center', va='center', 
                             fontsize=8, style='italic')
    
    def show_qkv_computation(self):
        """Show Query, Key, Value computation"""
        self.ax_attention.text(5, 1.5, "For each token, compute:", ha='center', fontsize=12, fontweight='bold')
        
        qkv_info = [
            ("Query (Q)", "What am I looking for?", "lightcoral"),
            ("Key (K)", "What do I represent?", "lightgreen"), 
            ("Value (V)", "What information do I carry?", "lightyellow")
        ]
        
        y_positions = [1.2, 1.0, 0.8]
        for (name, desc, color), y_pos in zip(qkv_info, y_positions):
            box = FancyBboxPatch((1, y_pos-0.08), 8, 0.15,
                               boxstyle="round,pad=0.02",
                               facecolor=color, alpha=0.7)
            self.ax_attention.add_patch(box)
            self.ax_attention.text(5, y_pos, f"{name}: {desc}", ha='center', va='center', fontsize=10)
    
    def show_attention_computation(self):
        """Show attention score computation"""
        self.ax_attention.text(5, 1.5, "Attention Score = softmax(Q·K^T / √d_k)", 
                             ha='center', fontsize=12, fontweight='bold')
        
        steps = [
            "1. Dot product: Query × Key^T",
            "2. Scale by √d_k (dimension of key vectors)",
            "3. Apply softmax to get probabilities"
        ]
        
        for i, step in enumerate(steps):
            y_pos = 1.2 - i * 0.2
            self.ax_attention.text(5, y_pos, step, ha='center', va='center', fontsize=10)
    
    def show_attention_focus(self, focus_token):
        """Show attention weights for a specific token"""
        attention_weights = self.attention_matrix[focus_token]
        
        # Draw attention connections
        x_positions = np.linspace(1, 9, self.n_tokens)
        focus_x = x_positions[focus_token]
        
        for i, (weight, x_pos) in enumerate(zip(attention_weights, x_positions)):
            if weight > 0.1:  # Only show significant connections
                # Connection line with thickness proportional to attention
                line_width = weight * 10
                alpha = min(weight * 2, 1.0)
                
                self.ax_attention.plot([focus_x, x_pos], [1.5, 0.5], 
                                    linewidth=line_width, alpha=alpha, color='red')
                
                # Attention weight label
                mid_x = (focus_x + x_pos) / 2
                mid_y = 1.0
                self.ax_attention.text(mid_x, mid_y, f'{weight:.2f}', 
                                     ha='center', va='center', fontsize=8,
                                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Draw tokens
        for i, (token, x_pos) in enumerate(zip(self.tokens, x_positions)):
            color = 'orange' if i == focus_token else 'lightblue'
            size = 0.4 if i == focus_token else 0.3
            
            box = FancyBboxPatch((x_pos-size/2, 1.5-size/2), size, size,
                               boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='black')
            self.ax_attention.add_patch(box)
            self.ax_attention.text(x_pos, 1.5, token, ha='center', va='center', 
                                 fontweight='bold', fontsize=9)
    
    def show_attention_matrix(self, highlight_row=None):
        """Display the attention matrix as a heatmap"""
        im = self.ax_matrix.imshow(self.attention_matrix, cmap='Blues', aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=self.ax_matrix, shrink=0.8)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        # Set labels
        self.ax_matrix.set_xticks(range(self.n_tokens))
        self.ax_matrix.set_yticks(range(self.n_tokens))
        self.ax_matrix.set_xticklabels(self.tokens, rotation=45, ha='right')
        self.ax_matrix.set_yticklabels(self.tokens)
        
        self.ax_matrix.set_xlabel('Keys (attending to)')
        self.ax_matrix.set_ylabel('Queries (attending from)')
        self.ax_matrix.set_title('Attention Matrix')
        
        # Highlight specific row if requested
        if highlight_row is not None:
            for j in range(self.n_tokens):
                weight = self.attention_matrix[highlight_row, j]
                self.ax_matrix.text(j, highlight_row, f'{weight:.2f}', 
                                  ha='center', va='center', fontweight='bold',
                                  color='white' if weight > 0.5 else 'black')
        
        # Add grid
        self.ax_matrix.set_xticks(np.arange(-0.5, self.n_tokens, 1), minor=True)
        self.ax_matrix.set_yticks(np.arange(-0.5, self.n_tokens, 1), minor=True)
        self.ax_matrix.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    def show_full_attention(self):
        """Show the complete attention pattern"""
        x_positions = np.linspace(1, 9, self.n_tokens)
        
        # Draw all significant attention connections
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                weight = self.attention_matrix[i, j]
                if weight > 0.15:  # Only show significant connections
                    line_width = weight * 8
                    alpha = min(weight * 1.5, 0.8)
                    color = plt.cm.viridis(weight)
                    
                    self.ax_attention.plot([x_positions[i], x_positions[j]], 
                                        [1.5, 0.5], 
                                        linewidth=line_width, alpha=alpha, color=color)
        
        # Draw tokens
        for i, (token, x_pos) in enumerate(zip(self.tokens, x_positions)):
            box = FancyBboxPatch((x_pos-0.3, 1.4), 0.6, 0.2,
                               boxstyle="round,pad=0.05",
                               facecolor='lightblue', edgecolor='black')
            self.ax_attention.add_patch(box)
            self.ax_attention.text(x_pos, 1.5, token, ha='center', va='center', 
                                 fontweight='bold', fontsize=9)
    
    def show_output_representation(self):
        """Show the final attended representation"""
        self.ax_output.text(5, 1.5, "Output: Attended Representation", 
                          ha='center', fontsize=12, fontweight='bold')
        
        x_positions = np.linspace(1, 9, self.n_tokens)
        
        for i, (token, x_pos) in enumerate(zip(self.tokens, x_positions)):
            # Output representation (enriched with context)
            box = FancyBboxPatch((x_pos-0.3, 0.7), 0.6, 0.4,
                               boxstyle="round,pad=0.05",
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
            self.ax_output.add_patch(box)
            
            self.ax_output.text(x_pos, 0.9, token, ha='center', va='center', 
                              fontweight='bold', fontsize=10)
            self.ax_output.text(x_pos, 0.5, "enriched", ha='center', va='center', 
                              fontsize=8, style='italic')

def demonstrate_attention_mechanism():
    """Run the attention mechanism animation demo"""
    sample_sentences = [
        "The cat sat on the mat",
        "She loves reading books",
        "AI transforms how we work"
    ]
    
    print("🤖 Starting Transformer Attention Mechanism Demo!")
    print("=" * 60)
    
    for i, sentence in enumerate(sample_sentences):
        print(f"\nDemo {i+1}: Analyzing '{sentence}'")
        
        animator = AttentionAnimator(sentence)
        
        # Create animation
        total_frames = 4 + len(animator.tokens) + 1
        anim = animation.FuncAnimation(
            animator.fig,
            animator.animate_attention,
            frames=total_frames,
            interval=3000,  # 3 seconds per frame
            repeat=True,
            blit=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # Print attention analysis
        print(f"Tokens: {animator.tokens}")
        print("Key attention patterns:")
        for j, token in enumerate(animator.tokens):
            top_attention = np.argsort(animator.attention_matrix[j])[-3:][::-1]
            attention_info = [f"{animator.tokens[k]}({animator.attention_matrix[j,k]:.2f})" 
                            for k in top_attention]
            print(f"  {token} → {', '.join(attention_info)}")
        print("-" * 50)

def explain_attention_concepts():
    """Explain key attention mechanism concepts"""
    print("\n" + "="*60)
    print("🧠 ATTENTION MECHANISM CONCEPTS")
    print("="*60)
    print()
    print("🔍 WHAT IS ATTENTION?")
    print("   - Mechanism that allows models to focus on relevant parts")
    print("   - Each token can 'attend' to other tokens in the sequence")
    print("   - Attention weights determine how much focus each token gets")
    print()
    print("🔑 KEY COMPONENTS:")
    print("   • Query (Q): What information am I looking for?")
    print("   • Key (K): What information do I represent?") 
    print("   • Value (V): What information do I actually provide?")
    print()
    print("⚡ HOW IT WORKS:")
    print("   1. Compute Q, K, V for each token")
    print("   2. Calculate attention scores: Q·K^T")
    print("   3. Apply softmax to get attention weights")
    print("   4. Weighted sum of Values based on attention weights")
    print()
    print("🎯 WHY IT'S POWERFUL:")
    print("   - Captures long-range dependencies")
    print("   - Allows parallel computation")
    print("   - Provides interpretability")
    print("   - Foundation of transformer models (GPT, BERT, etc.)")
    print("="*60)

if __name__ == "__main__":
    explain_attention_concepts()
    demonstrate_attention_mechanism()
