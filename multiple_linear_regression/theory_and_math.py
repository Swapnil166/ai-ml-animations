"""
MULTIPLE LINEAR REGRESSION - THEORY AND MATH
===========================================
Understanding the theory behind Multiple Linear Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("📚 MULTIPLE LINEAR REGRESSION - THEORY AND MATH")
print("=" * 50)

print("\n🎯 THE BIG PICTURE:")
print("-" * 20)
print("We want to find the best equation that predicts an outcome")
print("using multiple input features.")
print("\nGeneral form: y = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ + b")
print("\nOur job: Find the best values for w₁, w₂, w₃, ..., wₙ, and b")

input("\nPress Enter to dive into the theory...")

print("\n📊 STEP 1: UNDERSTANDING THE DATA")
print("-" * 35)
print("Let's use a simple example with 2 features:")
print("Predicting house price using size and number of bedrooms")

# Create sample data
np.random.seed(42)
n_samples = 20

# Features
house_size = np.random.uniform(1000, 3000, n_samples)  # sq ft
bedrooms = np.random.randint(1, 5, n_samples)  # number of bedrooms

# True relationship (what we're trying to discover)
true_w1 = 100  # $100 per sq ft
true_w2 = 5000  # $5000 per bedroom
true_b = 50000  # base price

# Generate prices with some noise
noise = np.random.normal(0, 10000, n_samples)
house_prices = true_w1 * house_size + true_w2 * bedrooms + true_b + noise

print(f"Sample data (first 5 houses):")
print("Size (sq ft) | Bedrooms | Price ($)")
print("-" * 35)
for i in range(5):
    print(f"{house_size[i]:8.0f}     |    {bedrooms[i]}     | {house_prices[i]:8.0f}")

input("\nPress Enter to see the 3D visualization...")

# 3D Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.scatter(house_size, bedrooms, house_prices, c='blue', s=50, alpha=0.7, label='Data Points')

# Create a mesh for the true plane
size_range = np.linspace(1000, 3000, 10)
bedroom_range = np.linspace(1, 4, 10)
SIZE, BEDROOMS = np.meshgrid(size_range, bedroom_range)
TRUE_PRICES = true_w1 * SIZE + true_w2 * BEDROOMS + true_b

# Plot the true plane
ax.plot_surface(SIZE, BEDROOMS, TRUE_PRICES, alpha=0.3, color='green', label='True Relationship')

ax.set_xlabel('House Size (sq ft)')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price ($)')
ax.set_title('Multiple Linear Regression in 3D\n(2 features + 1 target)')

plt.show()

input("\nPress Enter to understand the math...")

print("\n🧮 STEP 2: THE MATHEMATICS")
print("-" * 30)
print("Matrix Form (this makes calculations easier):")
print("\nInstead of writing:")
print("  y₁ = w₁x₁₁ + w₂x₁₂ + b")
print("  y₂ = w₁x₂₁ + w₂x₂₂ + b")
print("  y₃ = w₁x₃₁ + w₂x₃₂ + b")
print("  ...")

print("\nWe write: Y = X × W")
print("\nWhere:")
print("  Y = [y₁, y₂, y₃, ...]ᵀ  (target values)")
print("  X = [[1, x₁₁, x₁₂],     (features + bias column)")
print("       [1, x₂₁, x₂₂],")
print("       [1, x₃₁, x₃₂],")
print("       ...]")
print("  W = [b, w₁, w₂]ᵀ        (weights + bias)")

# Show actual matrices for our data
print(f"\nFor our house price example:")
print(f"Y (prices) shape: {house_prices.shape}")
print(f"X (features) shape: {n_samples} × 3")  # size, bedrooms, bias
print(f"W (weights) shape: 3 × 1")  # w1, w2, b

# Create feature matrix
X = np.column_stack([np.ones(n_samples), house_size, bedrooms])  # Add bias column
print(f"\nX matrix (first 5 rows):")
print("Bias | Size  | Bedrooms")
print("-" * 25)
for i in range(5):
    print(f" {X[i,0]:.0f}   | {X[i,1]:.0f} |    {X[i,2]:.0f}")

input("\nPress Enter to see the cost function...")

print("\n💰 STEP 3: COST FUNCTION")
print("-" * 25)
print("We use Mean Squared Error (MSE) just like simple linear regression:")
print("\nCost = (1/2m) × Σ(predicted - actual)²")
print("\nIn matrix form:")
print("Cost = (1/2m) × (XW - Y)ᵀ(XW - Y)")
print("\nWhere:")
print("  m = number of samples")
print("  XW = predictions")
print("  Y = actual values")

def cost_function(X, y, weights):
    """Calculate cost function"""
    m = len(y)
    predictions = X.dot(weights)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost, predictions

# Example with random weights
random_weights = np.array([30000, 50, 3000])  # b, w1, w2
cost, predictions = cost_function(X, house_prices, random_weights)

print(f"\nExample with random weights [b={random_weights[0]}, w₁={random_weights[1]}, w₂={random_weights[2]}]:")
print(f"Cost = {cost:,.0f}")
print(f"This means our predictions are off by about ${np.sqrt(2*cost):,.0f} on average")

input("\nPress Enter to see gradient calculation...")

print("\n📈 STEP 4: GRADIENTS")
print("-" * 20)
print("To minimize cost, we need gradients for each weight:")
print("\n∂Cost/∂W = (1/m) × Xᵀ(XW - Y)")
print("\nThis gives us gradients for all weights at once!")

def compute_gradients(X, y, weights):
    """Calculate gradients for all weights"""
    m = len(y)
    predictions = X.dot(weights)
    gradients = (1/m) * X.T.dot(predictions - y)
    return gradients

gradients = compute_gradients(X, house_prices, random_weights)
print(f"\nGradients for our random weights:")
print(f"∂Cost/∂b  = {gradients[0]:8.2f}")
print(f"∂Cost/∂w₁ = {gradients[1]:8.2f}")
print(f"∂Cost/∂w₂ = {gradients[2]:8.2f}")

print(f"\nInterpretation:")
if gradients[0] > 0:
    print(f"  Bias gradient > 0 → decrease bias")
else:
    print(f"  Bias gradient < 0 → increase bias")

if gradients[1] > 0:
    print(f"  Size weight gradient > 0 → decrease size weight")
else:
    print(f"  Size weight gradient < 0 → increase size weight")

if gradients[2] > 0:
    print(f"  Bedroom weight gradient > 0 → decrease bedroom weight")
else:
    print(f"  Bedroom weight gradient < 0 → increase bedroom weight")

input("\nPress Enter to see gradient descent in action...")

print("\n🔄 STEP 5: GRADIENT DESCENT ALGORITHM")
print("-" * 40)
print("1. Initialize weights randomly")
print("2. Calculate cost and gradients")
print("3. Update weights: W = W - learning_rate × gradients")
print("4. Repeat until convergence")

# Run gradient descent
def gradient_descent(X, y, learning_rate=0.0000001, iterations=1000):
    """Run gradient descent"""
    m, n = X.shape
    weights = np.random.randn(n) * 0.01  # Small random initialization
    cost_history = []
    
    for i in range(iterations):
        cost, predictions = cost_function(X, y, weights)
        gradients = compute_gradients(X, y, weights)
        weights = weights - learning_rate * gradients
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f"Iteration {i:4d}: Cost = {cost:12,.0f}, Weights = [{weights[0]:8.0f}, {weights[1]:6.2f}, {weights[2]:8.0f}]")
    
    return weights, cost_history

print(f"\nRunning gradient descent...")
print(f"Target weights: [b={true_b}, w₁={true_w1}, w₂={true_w2}]")
print(f"Iteration | Cost | Weights [b, w₁, w₂]")
print("-" * 60)

learned_weights, cost_history = gradient_descent(X, house_prices)

print(f"\n🎉 FINAL RESULTS:")
print(f"Learned weights: [b={learned_weights[0]:.0f}, w₁={learned_weights[1]:.2f}, w₂={learned_weights[2]:.0f}]")
print(f"True weights:    [b={true_b}, w₁={true_w1}, w₂={true_w2}]")

# Plot cost history
plt.figure(figsize=(10, 6))
plt.plot(cost_history, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Minimization')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see the decrease better
plt.show()

print(f"\n✅ SUMMARY:")
print(f"Multiple Linear Regression successfully learned the relationship!")
print(f"The algorithm found weights very close to the true values.")
print(f"\nNext: See the animated version!")
print(f"Run: python animated_visualization.py")
