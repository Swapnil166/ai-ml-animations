"""
GRADIENT DESCENT IN LINEAR REGRESSION - EDUCATIONAL TUTORIAL
===========================================================

This script teaches you gradient descent step by step with clear explanations.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("GRADIENT DESCENT IN LINEAR REGRESSION - TUTORIAL")
print("=" * 60)

# STEP 1: Understanding the Problem
print("\n🎯 STEP 1: THE PROBLEM")
print("-" * 20)
print("We want to find the best line y = mx + b that fits our data")
print("Where:")
print("  m = slope (how steep the line is)")
print("  b = intercept (where line crosses y-axis)")

# Create sample data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])  # Close to y = 2x + 1

print(f"\nOur data points:")
for i in range(len(X)):
    print(f"  Point {i+1}: x={X[i]}, y={y[i]}")

input("\nPress Enter to continue...")

# STEP 2: Cost Function
print("\n📊 STEP 2: MEASURING HOW GOOD OUR LINE IS")
print("-" * 45)
print("We use Mean Squared Error (MSE) as our cost function:")
print("Cost = (1/2n) × Σ(predicted - actual)²")
print("\nWhy squared? Because:")
print("  - Penalizes large errors more than small ones")
print("  - Always positive (no negative errors canceling positive ones)")
print("  - Makes math easier (derivatives are cleaner)")

def cost_function(m, b, X, y):
    """Calculate the cost (Mean Squared Error)"""
    predictions = m * X + b
    errors = predictions - y
    cost = np.mean(errors ** 2) / 2
    return cost, predictions, errors

# Example with a bad line
m_bad, b_bad = 0.5, 2
cost_bad, pred_bad, errors_bad = cost_function(m_bad, b_bad, X, y)

print(f"\nExample with line y = {m_bad}x + {b_bad}:")
print("x | actual | predicted | error | error²")
print("-" * 40)
total_squared_error = 0
for i in range(len(X)):
    error_sq = errors_bad[i] ** 2
    total_squared_error += error_sq
    print(f"{X[i]} |   {y[i]:2.0f}   |    {pred_bad[i]:4.1f}    | {errors_bad[i]:5.1f} | {error_sq:6.2f}")

print(f"\nTotal squared error: {total_squared_error:.2f}")
print(f"Mean squared error: {total_squared_error/len(X):.2f}")
print(f"Cost function value: {cost_bad:.2f}")

input("\nPress Enter to continue...")

# STEP 3: The Gradient (Slope of Cost Function)
print("\n📈 STEP 3: UNDERSTANDING GRADIENTS")
print("-" * 35)
print("Gradient = slope of the cost function")
print("It tells us:")
print("  - Which direction to move parameters (m and b)")
print("  - How steep the cost function is at current point")
print("\nGradient formulas:")
print("  ∂Cost/∂m = (1/n) × Σ(predicted - actual) × x")
print("  ∂Cost/∂b = (1/n) × Σ(predicted - actual)")

def compute_gradients(m, b, X, y):
    """Calculate gradients for slope and intercept"""
    predictions = m * X + b
    errors = predictions - y
    dm = np.mean(errors * X)  # Gradient w.r.t. slope
    db = np.mean(errors)      # Gradient w.r.t. intercept
    return dm, db

# Calculate gradients for our bad line
dm, db = compute_gradients(m_bad, b_bad, X, y)

print(f"\nFor our line y = {m_bad}x + {b_bad}:")
print(f"Gradient w.r.t. slope (∂Cost/∂m): {dm:.3f}")
print(f"Gradient w.r.t. intercept (∂Cost/∂b): {db:.3f}")
print(f"\nInterpretation:")
if dm > 0:
    print(f"  - Slope gradient is positive → decrease m to reduce cost")
else:
    print(f"  - Slope gradient is negative → increase m to reduce cost")
    
if db > 0:
    print(f"  - Intercept gradient is positive → decrease b to reduce cost")
else:
    print(f"  - Intercept gradient is negative → increase b to reduce cost")

input("\nPress Enter to continue...")

# STEP 4: Learning Rate
print("\n⚡ STEP 4: LEARNING RATE")
print("-" * 25)
print("Learning rate controls how big steps we take:")
print("  - Too small: slow convergence")
print("  - Too large: might overshoot the minimum")
print("  - Just right: efficient convergence")

learning_rates = [0.01, 0.1, 0.5]
print(f"\nLet's see different learning rates with gradient dm = {dm:.3f}:")
for lr in learning_rates:
    new_m = m_bad - lr * dm
    print(f"  Learning rate {lr:4.2f}: m = {m_bad} - {lr} × {dm:.3f} = {new_m:.3f}")

input("\nPress Enter to continue...")

# STEP 5: The Algorithm in Action
print("\n🔄 STEP 5: GRADIENT DESCENT ALGORITHM")
print("-" * 40)
print("Algorithm steps:")
print("1. Start with random parameters (m, b)")
print("2. Calculate cost")
print("3. Calculate gradients")
print("4. Update parameters: m = m - learning_rate × ∂Cost/∂m")
print("                     b = b - learning_rate × ∂Cost/∂b")
print("5. Repeat until convergence")

# Run gradient descent
m, b = 0, 0  # Start with zeros
learning_rate = 0.1
history = {'m': [], 'b': [], 'cost': []}

print(f"\nStarting gradient descent:")
print(f"Initial: m = {m}, b = {b}")
print(f"Learning rate: {learning_rate}")
print("\nIteration | m      | b      | Cost   | dm     | db")
print("-" * 55)

for iteration in range(10):
    # Store history
    cost, _, _ = cost_function(m, b, X, y)
    history['m'].append(m)
    history['b'].append(b)
    history['cost'].append(cost)
    
    # Calculate gradients
    dm, db = compute_gradients(m, b, X, y)
    
    # Print current state
    print(f"{iteration:9d} | {m:6.3f} | {b:6.3f} | {cost:6.3f} | {dm:6.3f} | {db:6.3f}")
    
    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db
    
    # Check for convergence
    if abs(dm) < 0.001 and abs(db) < 0.001:
        print(f"\nConverged at iteration {iteration}!")
        break

# Final results
final_cost, final_pred, _ = cost_function(m, b, X, y)
print(f"\n🎉 FINAL RESULTS:")
print(f"Learned equation: y = {m:.3f}x + {b:.3f}")
print(f"Final cost: {final_cost:.6f}")

print(f"\nFinal predictions:")
for i in range(len(X)):
    print(f"  x={X[i]} → actual={y[i]}, predicted={final_pred[i]:.2f}")

input("\nPress Enter to see the visualization...")

# STEP 6: Visualization
print("\n📊 STEP 6: VISUALIZING THE RESULTS")
print("-" * 35)

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Final fit
ax1.scatter(X, y, color='blue', s=100, label='Data points', zorder=5)
ax1.plot(X, final_pred, 'r-', linewidth=3, label=f'Learned: y = {m:.2f}x + {b:.2f}')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Final Line Fit')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cost over iterations
ax2.plot(history['cost'], 'b-', linewidth=2, marker='o')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')
ax2.set_title('Cost Function Minimization')
ax2.grid(True, alpha=0.3)

# Plot 3: Parameter m over iterations
ax3.plot(history['m'], 'g-', linewidth=2, marker='s', label='Slope (m)')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Slope (m)')
ax3.set_title('Slope Parameter Evolution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Parameter b over iterations
ax4.plot(history['b'], 'orange', linewidth=2, marker='^', label='Intercept (b)')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Intercept (b)')
ax4.set_title('Intercept Parameter Evolution')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("SUMMARY: WHAT YOU LEARNED")
print("=" * 60)
print("✅ Linear regression finds the best line y = mx + b")
print("✅ Cost function (MSE) measures how good our line is")
print("✅ Gradients tell us which direction to adjust parameters")
print("✅ Learning rate controls step size")
print("✅ Gradient descent iteratively improves the line")
print("✅ Algorithm converges to optimal solution")
print("\nKey insight: Machine learning is optimization!")
print("We're just finding parameters that minimize a cost function.")
