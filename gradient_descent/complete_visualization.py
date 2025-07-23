"""
GRADIENT DESCENT - COMPLETE VISUALIZATION
=========================================
This shows everything together: the math, the process, and the results.
Perfect for understanding the complete picture!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch

class GradientDescentVisualizer:
    def __init__(self):
        # Simple dataset
        self.X = np.array([1, 2, 3, 4, 5])
        self.y = np.array([2, 4, 6, 8, 10])  # Perfect line: y = 2x
        
        # Parameters
        self.m = 0  # Start with slope = 0
        self.b = 0  # Start with intercept = 0
        self.learning_rate = 0.1
        
        # History for animation
        self.history = {
            'm': [], 'b': [], 'cost': [], 'predictions': [], 
            'errors': [], 'gradients_m': [], 'gradients_b': []
        }
        
        # Run gradient descent
        self.run_gradient_descent()
    
    def cost_function(self, m, b):
        """Calculate Mean Squared Error"""
        predictions = m * self.X + b
        errors = predictions - self.y
        cost = np.mean(errors ** 2) / 2
        return cost, predictions, errors
    
    def compute_gradients(self, m, b):
        """Calculate gradients"""
        predictions = m * self.X + b
        errors = predictions - self.y
        dm = np.mean(errors * self.X)
        db = np.mean(errors)
        return dm, db
    
    def run_gradient_descent(self):
        """Run gradient descent and store history"""
        current_m, current_b = self.m, self.b
        
        for i in range(50):
            # Calculate current state
            cost, predictions, errors = self.cost_function(current_m, current_b)
            dm, db = self.compute_gradients(current_m, current_b)
            
            # Store history
            self.history['m'].append(current_m)
            self.history['b'].append(current_b)
            self.history['cost'].append(cost)
            self.history['predictions'].append(predictions.copy())
            self.history['errors'].append(errors.copy())
            self.history['gradients_m'].append(dm)
            self.history['gradients_b'].append(db)
            
            # Update parameters
            current_m -= self.learning_rate * dm
            current_b -= self.learning_rate * db
            
            # Check convergence
            if abs(dm) < 0.001 and abs(db) < 0.001:
                break
    
    def create_comprehensive_plot(self, frame):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Current parameters
        current_m = self.history['m'][frame]
        current_b = self.history['b'][frame]
        current_cost = self.history['cost'][frame]
        current_predictions = self.history['predictions'][frame]
        current_errors = self.history['errors'][frame]
        current_dm = self.history['gradients_m'][frame]
        current_db = self.history['gradients_b'][frame]
        
        # 1. Main plot: Data and line fit
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.scatter(self.X, self.y, s=150, color='blue', zorder=5, label='Data Points')
        ax1.plot(self.X, current_predictions, 'r-', linewidth=3, 
                label=f'Current Line: y = {current_m:.2f}x + {current_b:.2f}')
        ax1.plot(self.X, 2*self.X, 'g--', alpha=0.7, label='Target: y = 2x')
        
        # Show errors as vertical lines
        for i in range(len(self.X)):
            ax1.plot([self.X[i], self.X[i]], [self.y[i], current_predictions[i]], 
                    'r--', alpha=0.7, linewidth=2)
            ax1.text(self.X[i] + 0.1, (self.y[i] + current_predictions[i])/2, 
                    f'{current_errors[i]:.1f}', fontsize=10, color='red')
        
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title(f'Iteration {frame}: Line Fitting', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 6)
        ax1.set_ylim(0, 12)
        
        # 2. Cost over time
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(self.history['cost'][:frame+1], 'b-', linewidth=2)
        ax2.scatter(frame, current_cost, color='red', s=100, zorder=5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cost (MSE)')
        ax2.set_title(f'Cost: {current_cost:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # 3. Error visualization
        ax3 = fig.add_subplot(gs[1, 0])
        colors = ['red' if e > 0 else 'blue' for e in current_errors]
        ax3.bar(range(len(current_errors)), current_errors, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Data Point')
        ax3.set_ylabel('Error')
        ax3.set_title('Prediction Errors')
        ax3.grid(True, alpha=0.3)
        
        # 4. Squared errors
        ax4 = fig.add_subplot(gs[1, 1])
        squared_errors = current_errors ** 2
        ax4.bar(range(len(squared_errors)), squared_errors, color='green', alpha=0.7)
        ax4.set_xlabel('Data Point')
        ax4.set_ylabel('Squared Error')
        ax4.set_title('Squared Errors')
        ax4.grid(True, alpha=0.3)
        
        # 5. Parameter evolution
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(self.history['m'][:frame+1], 'g-', linewidth=2, label='Slope (m)')
        ax5.plot(self.history['b'][:frame+1], 'orange', linewidth=2, label='Intercept (b)')
        ax5.axhline(y=2, color='g', linestyle='--', alpha=0.7, label='Target m=2')
        ax5.axhline(y=0, color='orange', linestyle='--', alpha=0.7, label='Target b=0')
        ax5.scatter(frame, current_m, color='green', s=100, zorder=5)
        ax5.scatter(frame, current_b, color='orange', s=100, zorder=5)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Parameter Value')
        ax5.set_title('Parameter Evolution')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # 6. Math explanation
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create text boxes with math
        math_text = f"""
CURRENT MATH STEP {frame}:

📊 Current Line: y = {current_m:.3f}x + {current_b:.3f}

🔢 Predictions: [{', '.join([f'{p:.1f}' for p in current_predictions])}]
❌ Errors: [{', '.join([f'{e:.1f}' for e in current_errors])}]
⬜ Squared Errors: [{', '.join([f'{e**2:.1f}' for e in current_errors])}]

💰 Cost (MSE) = (1/2n) × Σ(error²) = {current_cost:.3f}

📈 Gradients:
   ∂Cost/∂m = (1/n) × Σ(error × x) = {current_dm:.3f}
   ∂Cost/∂b = (1/n) × Σ(error) = {current_db:.3f}

🔄 Parameter Updates (Learning Rate = {self.learning_rate}):
   m_new = {current_m:.3f} - {self.learning_rate} × {current_dm:.3f} = {current_m - self.learning_rate * current_dm:.3f}
   b_new = {current_b:.3f} - {self.learning_rate} × {current_db:.3f} = {current_b - self.learning_rate * current_db:.3f}
        """
        
        ax6.text(0.05, 0.95, math_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle(f'🎯 GRADIENT DESCENT - COMPLETE VISUALIZATION (Step {frame})', 
                    fontsize=16, fontweight='bold')
        
        return fig
    
    def show_static_summary(self):
        """Show final summary"""
        final_frame = len(self.history['m']) - 1
        fig = self.create_comprehensive_plot(final_frame)
        plt.show()
        
        print("\n" + "="*80)
        print("🎉 GRADIENT DESCENT COMPLETE!")
        print("="*80)
        print(f"Final Results:")
        print(f"  Learned slope: {self.history['m'][-1]:.3f} (target: 2.000)")
        print(f"  Learned intercept: {self.history['b'][-1]:.3f} (target: 0.000)")
        print(f"  Final cost: {self.history['cost'][-1]:.6f}")
        print(f"  Iterations: {len(self.history['m'])}")
        
        print(f"\nFinal predictions vs actual:")
        final_pred = self.history['predictions'][-1]
        for i in range(len(self.X)):
            print(f"  x={self.X[i]} → actual={self.y[i]}, predicted={final_pred[i]:.2f}")

def main():
    print("🎯 GRADIENT DESCENT - COMPLETE VISUALIZATION")
    print("=" * 50)
    print("This shows the complete gradient descent process with:")
    print("✅ Visual line fitting")
    print("✅ Cost function minimization")
    print("✅ Error analysis")
    print("✅ Parameter evolution")
    print("✅ Step-by-step math")
    
    visualizer = GradientDescentVisualizer()
    
    choice = input("\nChoose visualization:\n1. Static final result\n2. Step-by-step animation\nEnter choice (1 or 2): ")
    
    if choice == "1":
        visualizer.show_static_summary()
    elif choice == "2":
        print("Creating animation... (this may take a moment)")
        
        def animate(frame):
            plt.clf()
            return visualizer.create_comprehensive_plot(frame)
        
        fig = plt.figure(figsize=(16, 12))
        anim = animation.FuncAnimation(fig, animate, frames=len(visualizer.history['m']), 
                                     interval=1000, repeat=True, blit=False)
        plt.show()
    else:
        print("Invalid choice, showing static result...")
        visualizer.show_static_summary()

if __name__ == "__main__":
    main()
