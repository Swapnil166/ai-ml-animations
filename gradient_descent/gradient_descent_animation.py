import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2 * X + 1 + np.random.normal(0, 2, 50)  # True relationship: y = 2x + 1 + noise

# Initialize parameters
m = 0  # slope (weight)
b = 0  # intercept (bias)
learning_rate = 0.01
n = len(X)

# Store history for animation
m_history = []
b_history = []
cost_history = []

def compute_cost(m, b, X, y):
    """Compute Mean Squared Error cost function"""
    predictions = m * X + b
    cost = (1/(2*n)) * np.sum((predictions - y)**2)
    return cost

def compute_gradients(m, b, X, y):
    """Compute gradients for slope and intercept"""
    predictions = m * X + b
    dm = (1/n) * np.sum((predictions - y) * X)
    db = (1/n) * np.sum(predictions - y)
    return dm, db

# Run gradient descent and store history
current_m, current_b = m, b
for i in range(200):
    # Store current parameters
    m_history.append(current_m)
    b_history.append(current_b)
    cost_history.append(compute_cost(current_m, current_b, X, y))
    
    # Compute gradients
    dm, db = compute_gradients(current_m, current_b, X, y)
    
    # Update parameters
    current_m -= learning_rate * dm
    current_b -= learning_rate * db

# Set up the figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Animation function
def animate(frame):
    # Clear both axes
    ax1.clear()
    ax2.clear()
    
    # Left plot: Data points and regression line
    ax1.scatter(X, y, alpha=0.6, color='blue', label='Data points')
    
    # Current regression line
    current_m = m_history[frame]
    current_b = b_history[frame]
    y_pred = current_m * X + current_b
    ax1.plot(X, y_pred, 'r-', linewidth=2, label=f'y = {current_m:.2f}x + {current_b:.2f}')
    
    # True line for comparison
    y_true = 2 * X + 1
    ax1.plot(X, y_true, 'g--', alpha=0.7, label='True line: y = 2x + 1')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title(f'Linear Regression - Iteration {frame}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-5, 25)
    
    # Right plot: Cost function over time
    ax2.plot(cost_history[:frame+1], 'b-', linewidth=2)
    ax2.scatter(frame, cost_history[frame], color='red', s=50, zorder=5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost (MSE)')
    ax2.set_title('Cost Function Minimization')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(cost_history))
    ax2.set_ylim(0, max(cost_history) * 1.1)
    
    # Add text with current parameters and cost
    fig.suptitle(f'Gradient Descent: m={current_m:.3f}, b={current_b:.3f}, Cost={cost_history[frame]:.3f}', 
                 fontsize=14, fontweight='bold')

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=len(m_history), 
                              interval=100, repeat=True, blit=False)

plt.tight_layout()
plt.show()

# Print final results
print(f"\nFinal Results after {len(m_history)} iterations:")
print(f"Learned slope (m): {m_history[-1]:.3f}")
print(f"Learned intercept (b): {b_history[-1]:.3f}")
print(f"True slope: 2.000")
print(f"True intercept: 1.000")
print(f"Final cost: {cost_history[-1]:.3f}")
