"""
MULTIPLE LINEAR REGRESSION - ANIMATED VISUALIZATION
==================================================
Watch Multiple Linear Regression learn in real-time!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class MultipleLinearRegressionAnimator:
    def __init__(self):
        # Generate sample data
        np.random.seed(42)
        self.n_samples = 30
        
        # Features: house size and bedrooms
        self.house_size = np.random.uniform(1000, 3000, self.n_samples)
        self.bedrooms = np.random.randint(1, 5, self.n_samples)
        
        # True relationship
        self.true_w1 = 100   # $100 per sq ft
        self.true_w2 = 5000  # $5000 per bedroom
        self.true_b = 50000  # base price
        
        # Generate prices with noise
        noise = np.random.normal(0, 8000, self.n_samples)
        self.house_prices = (self.true_w1 * self.house_size + 
                           self.true_w2 * self.bedrooms + 
                           self.true_b + noise)
        
        # Create feature matrix
        self.X = np.column_stack([np.ones(self.n_samples), self.house_size, self.bedrooms])
        
        # Initialize parameters
        self.learning_rate = 0.0000001
        self.iterations = 200
        
        # Run gradient descent and store history
        self.weights_history, self.cost_history = self.run_gradient_descent()
    
    def cost_function(self, weights):
        """Calculate cost and predictions"""
        predictions = self.X.dot(weights)
        cost = (1/(2*self.n_samples)) * np.sum((predictions - self.house_prices)**2)
        return cost, predictions
    
    def compute_gradients(self, weights):
        """Calculate gradients"""
        predictions = self.X.dot(weights)
        gradients = (1/self.n_samples) * self.X.T.dot(predictions - self.house_prices)
        return gradients
    
    def run_gradient_descent(self):
        """Run gradient descent and store history"""
        weights = np.array([30000.0, 50.0, 3000.0])  # Start with bad weights
        weights_history = [weights.copy()]
        cost_history = []
        
        for i in range(self.iterations):
            cost, _ = self.cost_function(weights)
            gradients = self.compute_gradients(weights)
            weights = weights - self.learning_rate * gradients
            
            weights_history.append(weights.copy())
            cost_history.append(cost)
        
        return weights_history, cost_history
    
    def create_animation(self):
        """Create animated visualization"""
        fig = plt.figure(figsize=(16, 10))
        
        # Create subplots
        ax1 = plt.subplot(2, 3, (1, 2))  # Cost function
        ax2 = plt.subplot(2, 3, 3)       # Weights evolution
        ax3 = plt.subplot(2, 3, (4, 5))  # Predictions vs Actual
        ax4 = plt.subplot(2, 3, 6)       # Current parameters
        
        def animate(frame):
            # Clear all axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Current weights
            current_weights = self.weights_history[frame]
            current_cost = self.cost_history[frame] if frame > 0 else self.cost_history[0]
            _, current_predictions = self.cost_function(current_weights)
            
            # Plot 1: Cost function over time
            ax1.plot(self.cost_history[:frame+1], 'b-', linewidth=2)
            ax1.scatter(frame, current_cost, color='red', s=100, zorder=5)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Cost (MSE)')
            ax1.set_title(f'Cost Minimization (Iteration {frame})')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Plot 2: Weights evolution
            weights_array = np.array(self.weights_history[:frame+1])
            ax2.plot(weights_array[:, 0], 'g-', linewidth=2, label=f'Bias: {current_weights[0]:.0f}')
            ax2.plot(weights_array[:, 1], 'r-', linewidth=2, label=f'Size: {current_weights[1]:.2f}')
            ax2.plot(weights_array[:, 2], 'b-', linewidth=2, label=f'Bedrooms: {current_weights[2]:.0f}')
            
            # Target lines
            ax2.axhline(y=self.true_b, color='g', linestyle='--', alpha=0.7, label=f'Target Bias: {self.true_b}')
            ax2.axhline(y=self.true_w1, color='r', linestyle='--', alpha=0.7, label=f'Target Size: {self.true_w1}')
            ax2.axhline(y=self.true_w2, color='b', linestyle='--', alpha=0.7, label=f'Target Bedrooms: {self.true_w2}')
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Weight Value')
            ax2.set_title('Parameter Evolution')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Predictions vs Actual
            ax3.scatter(self.house_prices, current_predictions, alpha=0.6, s=50)
            
            # Perfect prediction line
            min_price = min(min(self.house_prices), min(current_predictions))
            max_price = max(max(self.house_prices), max(current_predictions))
            ax3.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Perfect Predictions')
            
            ax3.set_xlabel('Actual Price ($)')
            ax3.set_ylabel('Predicted Price ($)')
            ax3.set_title('Predictions vs Actual')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Calculate R²
            ss_res = np.sum((self.house_prices - current_predictions) ** 2)
            ss_tot = np.sum((self.house_prices - np.mean(self.house_prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            ax3.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
            
            # Plot 4: Current equation and info
            ax4.axis('off')
            ax4.text(0.1, 0.9, 'CURRENT EQUATION:', fontsize=14, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.1, 0.75, f'Price = {current_weights[1]:.2f} × Size', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.65, f'      + {current_weights[2]:.0f} × Bedrooms', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.55, f'      + {current_weights[0]:.0f}', fontsize=12, transform=ax4.transAxes)
            
            ax4.text(0.1, 0.4, 'TARGET EQUATION:', fontsize=14, fontweight='bold', color='green', transform=ax4.transAxes)
            ax4.text(0.1, 0.25, f'Price = {self.true_w1} × Size', fontsize=12, color='green', transform=ax4.transAxes)
            ax4.text(0.1, 0.15, f'      + {self.true_w2} × Bedrooms', fontsize=12, color='green', transform=ax4.transAxes)
            ax4.text(0.1, 0.05, f'      + {self.true_b}', fontsize=12, color='green', transform=ax4.transAxes)
            
            # Add cost info
            ax4.text(0.6, 0.9, f'Current Cost:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.6, 0.8, f'${current_cost:,.0f}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.6, 0.65, f'Avg Error:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.6, 0.55, f'${np.sqrt(2*current_cost):,.0f}', fontsize=12, transform=ax4.transAxes)
            
            plt.suptitle(f'Multiple Linear Regression Learning Process - Step {frame}/{len(self.weights_history)-1}', 
                        fontsize=16, fontweight='bold')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(self.weights_history), 
                                     interval=100, repeat=True, blit=False)
        
        plt.tight_layout()
        return anim
    
    def show_final_results(self):
        """Show final results"""
        final_weights = self.weights_history[-1]
        final_cost = self.cost_history[-1]
        
        print("\n" + "="*60)
        print("🎉 MULTIPLE LINEAR REGRESSION - FINAL RESULTS")
        print("="*60)
        
        print(f"\nLearned Equation:")
        print(f"Price = {final_weights[1]:.2f} × Size + {final_weights[2]:.0f} × Bedrooms + {final_weights[0]:.0f}")
        
        print(f"\nTarget Equation:")
        print(f"Price = {self.true_w1} × Size + {self.true_w2} × Bedrooms + {self.true_b}")
        
        print(f"\nAccuracy:")
        print(f"Size weight:     {final_weights[1]:.2f} vs {self.true_w1} (error: {abs(final_weights[1] - self.true_w1):.2f})")
        print(f"Bedroom weight:  {final_weights[2]:.0f} vs {self.true_w2} (error: {abs(final_weights[2] - self.true_w2):.0f})")
        print(f"Bias:            {final_weights[0]:.0f} vs {self.true_b} (error: {abs(final_weights[0] - self.true_b):.0f})")
        
        print(f"\nFinal Cost: ${final_cost:,.0f}")
        print(f"Average Prediction Error: ${np.sqrt(2*final_cost):,.0f}")
        
        # Test prediction
        test_size = 2000
        test_bedrooms = 3
        predicted_price = final_weights[1] * test_size + final_weights[2] * test_bedrooms + final_weights[0]
        true_price = self.true_w1 * test_size + self.true_w2 * test_bedrooms + self.true_b
        
        print(f"\nTest Prediction:")
        print(f"House: {test_size} sq ft, {test_bedrooms} bedrooms")
        print(f"Predicted price: ${predicted_price:,.0f}")
        print(f"True price:      ${true_price:,.0f}")
        print(f"Error:           ${abs(predicted_price - true_price):,.0f}")

def main():
    print("🎬 MULTIPLE LINEAR REGRESSION - ANIMATED VISUALIZATION")
    print("=" * 55)
    print("This animation shows how Multiple Linear Regression learns to predict")
    print("house prices using two features: size and number of bedrooms.")
    print("\nWhat you'll see:")
    print("✅ Cost function decreasing over time")
    print("✅ Weights converging to true values")
    print("✅ Predictions getting more accurate")
    print("✅ Real-time equation updates")
    
    animator = MultipleLinearRegressionAnimator()
    
    choice = input("\nChoose option:\n1. Show animation\n2. Show final results only\nEnter choice (1 or 2): ")
    
    if choice == "1":
        print("\nCreating animation... (this may take a moment)")
        anim = animator.create_animation()
        plt.show()
        animator.show_final_results()
    elif choice == "2":
        animator.show_final_results()
    else:
        print("Invalid choice, showing final results...")
        animator.show_final_results()

if __name__ == "__main__":
    main()
