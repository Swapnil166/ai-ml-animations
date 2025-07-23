"""
SIMPLE vs MULTIPLE LINEAR REGRESSION - COMPARISON
================================================
See the difference between using one feature vs multiple features!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

print("⚖️ SIMPLE vs MULTIPLE LINEAR REGRESSION")
print("=" * 45)
print("Let's see why using multiple features is better!")

# Generate realistic house data
np.random.seed(42)
n_houses = 100

# Features
house_size = np.random.uniform(1000, 3000, n_houses)
bedrooms = np.random.randint(1, 5, n_houses)
age = np.random.uniform(0, 30, n_houses)

# True relationship (complex)
true_price = (100 * house_size +     # $100 per sq ft
              8000 * bedrooms +      # $8000 per bedroom
              -300 * age +           # -$300 per year of age
              50000 +                # base price
              np.random.normal(0, 15000, n_houses))  # noise

print(f"Generated {n_houses} houses with:")
print(f"• Size: 1000-3000 sq ft")
print(f"• Bedrooms: 1-4")
print(f"• Age: 0-30 years")
print(f"• Price: Based on all three factors + noise")

input("\nPress Enter to compare models...")

# SIMPLE LINEAR REGRESSION (using only size)
def simple_linear_regression(x, y):
    """Simple linear regression: y = mx + b"""
    # Calculate slope and intercept
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    predictions = slope * x + intercept
    
    return slope, intercept, predictions

# MULTIPLE LINEAR REGRESSION
def multiple_linear_regression(X, y, learning_rate=0.0000001, iterations=1000):
    """Multiple linear regression using gradient descent"""
    m, n = X.shape
    weights = np.random.randn(n) * 0.01
    
    for i in range(iterations):
        predictions = X.dot(weights)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        gradients = (1/m) * X.T.dot(predictions - y)
        weights = weights - learning_rate * gradients
    
    predictions = X.dot(weights)
    return weights, predictions

# Fit Simple Linear Regression (size only)
simple_slope, simple_intercept, simple_predictions = simple_linear_regression(house_size, true_price)

# Fit Multiple Linear Regression (size + bedrooms + age)
X_multiple = np.column_stack([np.ones(n_houses), house_size, bedrooms, age])
multiple_weights, multiple_predictions = multiple_linear_regression(X_multiple, true_price)

# Calculate performance metrics
simple_r2 = r2_score(true_price, simple_predictions)
multiple_r2 = r2_score(true_price, multiple_predictions)

simple_mse = np.mean((simple_predictions - true_price)**2)
multiple_mse = np.mean((multiple_predictions - true_price)**2)

print(f"\n📊 RESULTS COMPARISON:")
print("=" * 30)
print(f"Simple Linear Regression (size only):")
print(f"  Equation: Price = {simple_slope:.2f} × Size + {simple_intercept:.0f}")
print(f"  R² Score: {simple_r2:.3f}")
print(f"  MSE: ${simple_mse:,.0f}")
print(f"  Avg Error: ${np.sqrt(simple_mse):,.0f}")

print(f"\nMultiple Linear Regression (size + bedrooms + age):")
print(f"  Equation: Price = {multiple_weights[1]:.2f} × Size + {multiple_weights[2]:.0f} × Bedrooms + {multiple_weights[3]:.2f} × Age + {multiple_weights[0]:.0f}")
print(f"  R² Score: {multiple_r2:.3f}")
print(f"  MSE: ${multiple_mse:,.0f}")
print(f"  Avg Error: ${np.sqrt(multiple_mse):,.0f}")

improvement = ((simple_mse - multiple_mse) / simple_mse) * 100
print(f"\n🎯 IMPROVEMENT:")
print(f"Multiple regression is {improvement:.1f}% more accurate!")

input("\nPress Enter to see visualizations...")

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Simple Linear Regression
ax1.scatter(house_size, true_price, alpha=0.6, s=30, label='Actual Prices')
ax1.plot(house_size, simple_predictions, 'r-', linewidth=2, label=f'Simple Model (R²={simple_r2:.3f})')
ax1.set_xlabel('House Size (sq ft)')
ax1.set_ylabel('Price ($)')
ax1.set_title('Simple Linear Regression\n(Size Only)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Multiple Linear Regression - Predictions vs Actual
ax2.scatter(true_price, multiple_predictions, alpha=0.6, s=30)
min_price = min(min(true_price), min(multiple_predictions))
max_price = max(max(true_price), max(multiple_predictions))
ax2.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Perfect Predictions')
ax2.set_xlabel('Actual Price ($)')
ax2.set_ylabel('Predicted Price ($)')
ax2.set_title(f'Multiple Linear Regression\n(R²={multiple_r2:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals comparison
simple_residuals = simple_predictions - true_price
multiple_residuals = multiple_predictions - true_price

ax3.scatter(range(len(simple_residuals)), simple_residuals, alpha=0.6, s=20, color='red', label='Simple Model')
ax3.scatter(range(len(multiple_residuals)), multiple_residuals, alpha=0.6, s=20, color='blue', label='Multiple Model')
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax3.set_xlabel('House Index')
ax3.set_ylabel('Residual (Predicted - Actual)')
ax3.set_title('Prediction Errors Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Feature importance (for multiple regression)
feature_names = ['Bias', 'Size', 'Bedrooms', 'Age']
colors = ['gray', 'blue', 'green', 'orange']
bars = ax4.bar(feature_names, multiple_weights, color=colors, alpha=0.7)
ax4.set_ylabel('Weight Value')
ax4.set_title('Feature Importance\n(Multiple Linear Regression)')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, weight in zip(bars, multiple_weights):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + (height*0.05 if height > 0 else height*0.05),
             f'{weight:.0f}', ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()
plt.show()

print(f"\n🔍 DETAILED ANALYSIS:")
print("=" * 25)

# Show some specific examples
print(f"\nExample predictions for 3 houses:")
print("House | Size | Bedrooms | Age | Actual | Simple | Multiple | Simple Error | Multiple Error")
print("-" * 95)

for i in range(3):
    idx = i * 30  # Pick every 30th house for variety
    actual = true_price[idx]
    simple_pred = simple_predictions[idx]
    multiple_pred = multiple_predictions[idx]
    simple_error = abs(simple_pred - actual)
    multiple_error = abs(multiple_pred - actual)
    
    print(f"  {idx+1:2d}  | {house_size[idx]:4.0f} |    {bedrooms[idx]}     | {age[idx]:2.0f}  | {actual:6.0f} | {simple_pred:6.0f} |  {multiple_pred:6.0f}   |    {simple_error:6.0f}    |     {multiple_error:6.0f}")

print(f"\n💡 KEY INSIGHTS:")
print("=" * 20)
print("✅ Multiple Linear Regression captures more complex relationships")
print("✅ Using more relevant features improves accuracy")
print("✅ Simple models are easier to interpret but less accurate")
print("✅ Multiple models can reveal feature importance")

print(f"\n🎯 WHEN TO USE EACH:")
print("=" * 25)
print("Simple Linear Regression:")
print("  • When you have only one important feature")
print("  • When you need easy interpretation")
print("  • For quick initial analysis")

print("\nMultiple Linear Regression:")
print("  • When multiple factors affect the outcome")
print("  • When accuracy is more important than simplicity")
print("  • For real-world problems (most cases!)")

print(f"\n🚀 NEXT STEPS:")
print("Try the animated visualization to see the learning process!")
print("Run: python animated_visualization.py")
