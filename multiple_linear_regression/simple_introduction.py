"""
MULTIPLE LINEAR REGRESSION - SIMPLE INTRODUCTION
===============================================
Let's understand Multiple Linear Regression using simple examples!
"""

import numpy as np
import matplotlib.pyplot as plt

print("🏠 MULTIPLE LINEAR REGRESSION - SIMPLE INTRODUCTION")
print("=" * 55)

print("\n🤔 WHAT IS MULTIPLE LINEAR REGRESSION?")
print("-" * 40)
print("Think of it like this:")
print("• Simple Linear Regression: Predicting house price using ONLY house size")
print("• Multiple Linear Regression: Predicting house price using house size AND number of bedrooms AND age")
print("\nIt's like having multiple factors that affect the outcome!")

input("\nPress Enter to see examples...")

print("\n📊 REAL-WORLD EXAMPLES:")
print("-" * 25)
print("1. 🏠 House Price Prediction:")
print("   Price = f(size, bedrooms, age, location)")
print("\n2. 📚 Student Grade Prediction:")
print("   Grade = f(study_hours, attendance, previous_grades)")
print("\n3. 🚗 Car Price Prediction:")
print("   Price = f(mileage, age, brand, engine_size)")
print("\n4. 💰 Salary Prediction:")
print("   Salary = f(experience, education, skills, location)")

input("\nPress Enter to see the math...")

print("\n🧮 THE MATH (SIMPLE VERSION):")
print("-" * 30)
print("Simple Linear Regression:")
print("  y = mx + b")
print("  (one input, one output)")
print("\nMultiple Linear Regression:")
print("  y = w₁x₁ + w₂x₂ + w₃x₃ + ... + b")
print("  (multiple inputs, one output)")
print("\nWhere:")
print("  y = what we want to predict (house price)")
print("  x₁, x₂, x₃ = different features (size, bedrooms, age)")
print("  w₁, w₂, w₃ = weights (how important each feature is)")
print("  b = bias (base value)")

input("\nPress Enter to see a concrete example...")

print("\n🏠 CONCRETE EXAMPLE: HOUSE PRICE PREDICTION")
print("-" * 45)
print("Let's say we want to predict house prices using:")
print("  x₁ = House size (in sq ft)")
print("  x₂ = Number of bedrooms")
print("  x₃ = Age of house (in years)")

print("\nOur equation might be:")
print("  Price = 100×Size + 5000×Bedrooms - 500×Age + 50000")
print("\nThis means:")
print("  • Each sq ft adds $100 to price")
print("  • Each bedroom adds $5,000 to price")
print("  • Each year of age reduces price by $500")
print("  • Base price is $50,000")

# Example calculation
size = 2000
bedrooms = 3
age = 10

predicted_price = 100 * size + 5000 * bedrooms - 500 * age + 50000

print(f"\nExample house:")
print(f"  Size: {size} sq ft")
print(f"  Bedrooms: {bedrooms}")
print(f"  Age: {age} years")
print(f"\nPredicted price:")
print(f"  = 100×{size} + 5000×{bedrooms} - 500×{age} + 50000")
print(f"  = {100*size} + {5000*bedrooms} + {-500*age} + 50000")
print(f"  = ${predicted_price:,}")

input("\nPress Enter to see the difference from simple regression...")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Simple Linear Regression (2D)
x_simple = np.array([1000, 1500, 2000, 2500, 3000])
y_simple = np.array([150000, 200000, 250000, 300000, 350000])

ax1.scatter(x_simple, y_simple, s=100, color='blue', alpha=0.7)
ax1.plot(x_simple, 100 * x_simple + 50000, 'r-', linewidth=2, label='Price = 100×Size + 50000')
ax1.set_xlabel('House Size (sq ft)')
ax1.set_ylabel('Price ($)')
ax1.set_title('Simple Linear Regression\n(1 input: Size only)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Multiple Linear Regression visualization (showing the concept)
ax2.text(0.5, 0.8, 'Multiple Linear Regression', ha='center', va='center', 
         transform=ax2.transAxes, fontsize=16, fontweight='bold')

# Draw input features
features = ['House Size', 'Bedrooms', 'Age', 'Location']
weights = ['w₁ = 100', 'w₂ = 5000', 'w₃ = -500', 'w₄ = 20000']

for i, (feature, weight) in enumerate(zip(features, weights)):
    y_pos = 0.6 - i * 0.1
    ax2.text(0.1, y_pos, feature, transform=ax2.transAxes, fontsize=12)
    ax2.text(0.4, y_pos, weight, transform=ax2.transAxes, fontsize=12, color='red')
    ax2.arrow(0.35, y_pos, 0.25, 0, transform=ax2.transAxes, 
              head_width=0.02, head_length=0.03, fc='blue', ec='blue')

ax2.text(0.7, 0.4, 'House\nPrice', ha='center', va='center', 
         transform=ax2.transAxes, fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))

ax2.text(0.5, 0.1, 'Price = w₁×Size + w₂×Bedrooms + w₃×Age + w₄×Location + b', 
         ha='center', transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('Multiple Linear Regression\n(Multiple inputs)')

plt.tight_layout()
plt.show()

print("\n🎯 KEY DIFFERENCES:")
print("-" * 20)
print("Simple Linear Regression:")
print("  ✅ Easy to visualize (2D line)")
print("  ❌ Limited - only uses one feature")
print("  ❌ Less accurate for complex problems")

print("\nMultiple Linear Regression:")
print("  ✅ Uses multiple features")
print("  ✅ More accurate predictions")
print("  ✅ Captures complex relationships")
print("  ❌ Harder to visualize (multi-dimensional)")

print("\n🚀 NEXT STEPS:")
print("Now let's see how to find the best weights using gradient descent!")
print("Run: python theory_and_math.py")
