"""
GRADIENT DESCENT - VISUAL MATH TUTORIAL
=======================================
This tutorial explains gradient descent using simple visuals and analogies.
Perfect for people who find math challenging!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import time

def create_title_slide(title, subtitle=""):
    """Create a title slide"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.5, 0.6, title, fontsize=24, fontweight='bold', 
            ha='center', va='center', transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.4, subtitle, fontsize=16, 
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax

def wait_for_user():
    """Wait for user to press Enter"""
    input("\nPress Enter to continue...")
    plt.close('all')

print("🎓 GRADIENT DESCENT - VISUAL MATH TUTORIAL")
print("=" * 50)
print("This tutorial will teach you gradient descent using:")
print("✅ Simple analogies (like rolling a ball down a hill)")
print("✅ Visual examples (lots of pictures!)")
print("✅ Step-by-step breakdowns")
print("✅ No scary math formulas (we'll build them together)")
print("\nLet's start!")

input("\nPress Enter to begin the visual journey...")

# SLIDE 1: The Big Picture
fig, ax = create_title_slide(
    "🎯 THE BIG PICTURE", 
    "What is Gradient Descent?"
)
ax.text(0.5, 0.25, "Imagine you're blindfolded on a hill and want to reach the bottom.\nGradient Descent is like feeling the slope and taking steps downhill.", 
        fontsize=14, ha='center', va='center', transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
plt.show()
wait_for_user()

# SLIDE 2: The Hill Analogy
fig, ax = plt.subplots(figsize=(12, 8))
x = np.linspace(-3, 3, 100)
y = x**2 + 1  # Simple parabola (hill shape)

ax.plot(x, y, 'b-', linewidth=3, label='Cost Function (Our Hill)')
ax.fill_between(x, y, alpha=0.3, color='green')

# Add a person at different positions
positions = [-2, 0, 2]
colors = ['red', 'orange', 'purple']
labels = ['Start Here', 'Goal!', 'Or Here']

for i, (pos, color, label) in enumerate(zip(positions, colors, labels)):
    ax.scatter(pos, pos**2 + 1, s=200, color=color, zorder=5)
    ax.annotate(label, (pos, pos**2 + 1), xytext=(pos, pos**2 + 3),
                ha='center', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

ax.set_xlabel('Parameter Value (where you are on the hill)', fontsize=14)
ax.set_ylabel('Cost (height of the hill)', fontsize=14)
ax.set_title('🏔️ The Hill Analogy: Finding the Bottom', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

# Add explanation text
ax.text(0.02, 0.98, 
        "Key Insight: We want to find the LOWEST point (minimum cost)\n" +
        "Just like rolling a ball down a hill!", 
        transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.show()
wait_for_user()

# SLIDE 3: What is a Slope?
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: Positive slope
x1 = np.linspace(0, 5, 100)
y1 = 2 * x1 + 1
ax1.plot(x1, y1, 'r-', linewidth=3)
ax1.arrow(2, 5, 1, 2, head_width=0.2, head_length=0.3, fc='red', ec='red')
ax1.set_title('📈 Positive Slope = Going UP', fontsize=14, fontweight='bold')
ax1.text(2.5, 3, 'Slope = +2\n(up 2, right 1)', fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="pink"))
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Right: Negative slope
x2 = np.linspace(0, 5, 100)
y2 = -2 * x2 + 10
ax2.plot(x2, y2, 'b-', linewidth=3)
ax2.arrow(2, 6, 1, -2, head_width=0.2, head_length=0.3, fc='blue', ec='blue')
ax2.set_title('📉 Negative Slope = Going DOWN', fontsize=14, fontweight='bold')
ax2.text(2.5, 8, 'Slope = -2\n(down 2, right 1)', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.tight_layout()
plt.show()
wait_for_user()

# SLIDE 4: Gradient = Slope of the Hill
fig, ax = plt.subplots(figsize=(12, 8))
x = np.linspace(-3, 3, 100)
y = x**2 + 1

ax.plot(x, y, 'b-', linewidth=3, label='Cost Function')

# Show gradients at different points
points = [-2, -1, 0, 1, 2]
for point in points:
    gradient = 2 * point  # derivative of x^2 is 2x
    y_point = point**2 + 1
    
    # Draw tangent line
    tangent_x = np.linspace(point-0.5, point+0.5, 10)
    tangent_y = y_point + gradient * (tangent_x - point)
    ax.plot(tangent_x, tangent_y, 'r--', alpha=0.7, linewidth=2)
    
    # Mark the point
    ax.scatter(point, y_point, s=100, color='red', zorder=5)
    
    # Add gradient value
    ax.annotate(f'Gradient = {gradient:.1f}', 
                (point, y_point), xytext=(point, y_point + 2),
                ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

ax.set_xlabel('Parameter Value', fontsize=14)
ax.set_ylabel('Cost', fontsize=14)
ax.set_title('🎯 Gradient = Slope of the Cost Function at Each Point', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add explanation
ax.text(0.02, 0.98, 
        "Red dashed lines = slopes (gradients) at different points\n" +
        "Negative gradient = go RIGHT to go downhill\n" +
        "Positive gradient = go LEFT to go downhill", 
        transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.show()
wait_for_user()

# SLIDE 5: The Learning Rate
fig, ax = plt.subplots(figsize=(12, 8))
x = np.linspace(-3, 3, 100)
y = x**2 + 1
ax.plot(x, y, 'b-', linewidth=3, label='Cost Function')

# Show different step sizes
start_point = 2
learning_rates = [0.1, 0.5, 1.0]
colors = ['green', 'orange', 'red']
labels = ['Small Steps (LR=0.1)', 'Medium Steps (LR=0.5)', 'Big Steps (LR=1.0)']

for lr, color, label in zip(learning_rates, colors, labels):
    current_x = start_point
    path_x, path_y = [current_x], [current_x**2 + 1]
    
    for step in range(5):
        gradient = 2 * current_x
        current_x = current_x - lr * gradient
        path_x.append(current_x)
        path_y.append(current_x**2 + 1)
    
    ax.plot(path_x, path_y, 'o-', color=color, linewidth=2, 
            markersize=8, label=label, alpha=0.8)

ax.set_xlabel('Parameter Value', fontsize=14)
ax.set_ylabel('Cost', fontsize=14)
ax.set_title('🚶 Learning Rate = Step Size', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add explanation
ax.text(0.02, 0.98, 
        "Learning Rate controls how big steps you take:\n" +
        "🟢 Small = Slow but safe\n" +
        "🟠 Medium = Good balance\n" +
        "🔴 Big = Fast but might overshoot", 
        transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))

plt.tight_layout()
plt.show()
wait_for_user()

# SLIDE 6: Linear Regression Problem
fig, ax = plt.subplots(figsize=(12, 8))

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

ax.scatter(X, y, s=150, color='blue', zorder=5, label='Data Points')

# Show different possible lines
lines = [
    (0.5, 1, 'red', 'Bad Line: y = 0.5x + 1'),
    (1.5, 0.5, 'orange', 'Better Line: y = 1.5x + 0.5'),
    (2, 0, 'green', 'Perfect Line: y = 2x + 0')
]

x_line = np.linspace(0, 6, 100)
for slope, intercept, color, label in lines:
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linewidth=2, label=label)

ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_title('🎯 Linear Regression: Finding the Best Line', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 6)
ax.set_ylim(0, 12)

# Add explanation
ax.text(0.02, 0.98, 
        "Goal: Find the line y = mx + b that best fits the blue dots\n" +
        "We need to find the best values for:\n" +
        "• m (slope) - how steep the line is\n" +
        "• b (intercept) - where line crosses y-axis", 
        transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

plt.tight_layout()
plt.show()
wait_for_user()

# SLIDE 7: What is Error?
fig, ax = plt.subplots(figsize=(12, 8))

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
ax.scatter(X, y, s=150, color='blue', zorder=5, label='Actual Data')

# Bad line
m_bad, b_bad = 1, 1
y_pred_bad = m_bad * X + b_bad
ax.plot(X, y_pred_bad, 'r-', linewidth=3, label=f'Bad Line: y = {m_bad}x + {b_bad}')

# Show errors as vertical lines
for i in range(len(X)):
    error = y_pred_bad[i] - y[i]
    ax.plot([X[i], X[i]], [y[i], y_pred_bad[i]], 'r--', linewidth=2, alpha=0.7)
    ax.text(X[i] + 0.1, (y[i] + y_pred_bad[i])/2, f'Error: {error:.0f}', 
            fontsize=10, color='red', fontweight='bold')

ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_title('❌ Understanding Errors', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add explanation
ax.text(0.02, 0.98, 
        "Error = Predicted - Actual\n" +
        "Red dashed lines show errors\n" +
        "We want to minimize these errors!", 
        transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose"))

plt.tight_layout()
plt.show()
wait_for_user()

# SLIDE 8: Why Square the Errors?
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

errors = np.array([-3, -1, 2, 1, -2])
squared_errors = errors ** 2

# Left: Regular errors
ax1.bar(range(len(errors)), errors, color=['red' if e < 0 else 'blue' for e in errors])
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax1.set_title('😕 Regular Errors Can Cancel Out', fontsize=14, fontweight='bold')
ax1.set_ylabel('Error')
ax1.set_xlabel('Data Point')
ax1.text(0.5, 0.95, f'Sum of errors: {np.sum(errors):.0f}\n(Positive and negative cancel!)', 
         transform=ax1.transAxes, ha='center', va='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

# Right: Squared errors
ax2.bar(range(len(squared_errors)), squared_errors, color='green', alpha=0.7)
ax2.set_title('😊 Squared Errors Are Always Positive', fontsize=14, fontweight='bold')
ax2.set_ylabel('Squared Error')
ax2.set_xlabel('Data Point')
ax2.text(0.5, 0.95, f'Sum of squared errors: {np.sum(squared_errors):.0f}\n(All positive - no canceling!)', 
         transform=ax2.transAxes, ha='center', va='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

plt.tight_layout()
plt.show()
wait_for_user()

print("\n🎉 CONGRATULATIONS!")
print("You've completed the visual math tutorial!")
print("\nWhat you learned:")
print("✅ Gradient descent is like rolling a ball down a hill")
print("✅ Gradient = slope = direction to move")
print("✅ Learning rate = step size")
print("✅ We square errors so they don't cancel out")
print("✅ Goal: Find parameters that minimize cost")

print("\nNext: Run the interactive tutorial to see it all in action!")
print("Command: python learn_gradient_descent.py")
