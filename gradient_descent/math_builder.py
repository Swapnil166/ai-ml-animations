"""
GRADIENT DESCENT - INTERACTIVE MATH BUILDER
===========================================
This builds up the math formulas step by step with visual examples.
"""

import numpy as np
import matplotlib.pyplot as plt

def show_formula_step(step_num, title, formula, explanation, example=None):
    """Show a formula step with explanation"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print('='*60)
    print(f"📝 FORMULA: {formula}")
    print(f"💡 EXPLANATION: {explanation}")
    if example:
        print(f"🔢 EXAMPLE: {example}")
    print('='*60)

def interactive_math_builder():
    """Build up gradient descent math step by step"""
    
    print("🧮 GRADIENT DESCENT - MATH BUILDER")
    print("Let's build the math formulas step by step!")
    print("We'll use simple numbers so you can follow along.")
    
    input("\nPress Enter to start building the math...")
    
    # Our simple dataset
    X = np.array([1, 2, 3])
    y = np.array([2, 4, 6])  # Perfect line: y = 2x
    
    print(f"\n📊 OUR DATA:")
    print("X = [1, 2, 3]")
    print("Y = [2, 4, 6]")
    print("(Notice: Y is always 2 times X, so perfect line is y = 2x)")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Prediction Formula
    show_formula_step(
        1, 
        "MAKING PREDICTIONS",
        "prediction = m × x + b",
        "To predict y, we multiply x by slope (m) and add intercept (b)",
        "If m=1, b=0.5, and x=2, then prediction = 1×2 + 0.5 = 2.5"
    )
    
    # Let's use a bad line to show errors
    m, b = 1, 0.5  # Bad parameters
    predictions = m * X + b
    
    print(f"\nUsing our bad line y = {m}x + {b}:")
    for i in range(len(X)):
        print(f"  x={X[i]} → prediction = {m}×{X[i]} + {b} = {predictions[i]}")
    
    input("\nPress Enter for next step...")
    
    # Step 2: Error Calculation
    show_formula_step(
        2,
        "CALCULATING ERRORS",
        "error = prediction - actual",
        "Error tells us how far off our prediction is from the real value",
        "If prediction=2.5 and actual=2, then error = 2.5 - 2 = 0.5"
    )
    
    errors = predictions - y
    print(f"\nErrors for our bad line:")
    for i in range(len(X)):
        print(f"  Point {i+1}: error = {predictions[i]} - {y[i]} = {errors[i]}")
    
    input("\nPress Enter for next step...")
    
    # Step 3: Squared Errors
    show_formula_step(
        3,
        "SQUARING THE ERRORS",
        "squared_error = error²",
        "We square errors to make them all positive and penalize big errors more",
        "If error = -0.5, then squared_error = (-0.5)² = 0.25"
    )
    
    squared_errors = errors ** 2
    print(f"\nSquared errors:")
    for i in range(len(X)):
        print(f"  Point {i+1}: ({errors[i]})² = {squared_errors[i]}")
    
    input("\nPress Enter for next step...")
    
    # Step 4: Mean Squared Error
    show_formula_step(
        4,
        "MEAN SQUARED ERROR (COST FUNCTION)",
        "MSE = (1/n) × Σ(squared_errors)",
        "Average of all squared errors. This is our cost function to minimize",
        f"MSE = (1/3) × ({squared_errors[0]} + {squared_errors[1]} + {squared_errors[2]}) = {np.mean(squared_errors):.3f}"
    )
    
    mse = np.mean(squared_errors)
    print(f"\nOur cost (MSE) = {mse:.3f}")
    print("Goal: Make this number as small as possible!")
    
    input("\nPress Enter for next step...")
    
    # Step 5: Gradient with respect to slope (m)
    show_formula_step(
        5,
        "GRADIENT FOR SLOPE (∂Cost/∂m)",
        "∂Cost/∂m = (1/n) × Σ(error × x)",
        "This tells us how much the cost changes when we change the slope",
        "If we increase m slightly, how much does the cost change?"
    )
    
    # Calculate gradient for slope
    dm = np.mean(errors * X)
    print(f"\nCalculating gradient for slope:")
    print("∂Cost/∂m = (1/n) × Σ(error × x)")
    print(f"         = (1/3) × [({errors[0]}×{X[0]}) + ({errors[1]}×{X[1]}) + ({errors[2]}×{X[2]})]")
    print(f"         = (1/3) × [{errors[0]*X[0]} + {errors[1]*X[1]} + {errors[2]*X[2]}]")
    print(f"         = (1/3) × {np.sum(errors * X)}")
    print(f"         = {dm:.3f}")
    
    print(f"\nInterpretation: Since gradient = {dm:.3f} > 0,")
    print("we should DECREASE m to reduce cost")
    
    input("\nPress Enter for next step...")
    
    # Step 6: Gradient with respect to intercept (b)
    show_formula_step(
        6,
        "GRADIENT FOR INTERCEPT (∂Cost/∂b)",
        "∂Cost/∂b = (1/n) × Σ(error)",
        "This tells us how much the cost changes when we change the intercept",
        "If we increase b slightly, how much does the cost change?"
    )
    
    # Calculate gradient for intercept
    db = np.mean(errors)
    print(f"\nCalculating gradient for intercept:")
    print("∂Cost/∂b = (1/n) × Σ(error)")
    print(f"         = (1/3) × [{errors[0]} + {errors[1]} + {errors[2]}]")
    print(f"         = (1/3) × {np.sum(errors)}")
    print(f"         = {db:.3f}")
    
    print(f"\nInterpretation: Since gradient = {db:.3f} < 0,")
    print("we should INCREASE b to reduce cost")
    
    input("\nPress Enter for next step...")
    
    # Step 7: Parameter Updates
    show_formula_step(
        7,
        "UPDATING PARAMETERS",
        "m_new = m_old - learning_rate × ∂Cost/∂m\nb_new = b_old - learning_rate × ∂Cost/∂b",
        "Move parameters in opposite direction of gradient to reduce cost",
        "If gradient is positive, subtract it. If gradient is negative, subtracting makes it positive (so we add)"
    )
    
    learning_rate = 0.1
    m_new = m - learning_rate * dm
    b_new = b - learning_rate * db
    
    print(f"\nUsing learning rate = {learning_rate}:")
    print(f"m_new = {m} - {learning_rate} × {dm:.3f} = {m_new:.3f}")
    print(f"b_new = {b} - {learning_rate} × {db:.3f} = {b_new:.3f}")
    
    print(f"\nOld line: y = {m}x + {b}")
    print(f"New line: y = {m_new:.3f}x + {b_new:.3f}")
    
    # Calculate new cost
    new_predictions = m_new * X + b_new
    new_errors = new_predictions - y
    new_mse = np.mean(new_errors ** 2)
    
    print(f"\nOld cost: {mse:.3f}")
    print(f"New cost: {new_mse:.3f}")
    print(f"Improvement: {mse - new_mse:.3f} (cost went down! 🎉)")
    
    input("\nPress Enter to see the visual comparison...")
    
    # Visual comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before
    ax1.scatter(X, y, s=100, color='blue', zorder=5, label='Data')
    ax1.plot(X, predictions, 'r-', linewidth=3, label=f'Old: y = {m}x + {b}')
    for i in range(len(X)):
        ax1.plot([X[i], X[i]], [y[i], predictions[i]], 'r--', alpha=0.7)
    ax1.set_title(f'BEFORE: Cost = {mse:.3f}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 8)
    
    # After
    ax2.scatter(X, y, s=100, color='blue', zorder=5, label='Data')
    ax2.plot(X, new_predictions, 'g-', linewidth=3, label=f'New: y = {m_new:.2f}x + {b_new:.2f}')
    for i in range(len(X)):
        ax2.plot([X[i], X[i]], [y[i], new_predictions[i]], 'g--', alpha=0.7)
    ax2.set_title(f'AFTER: Cost = {new_mse:.3f}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 8)
    
    plt.tight_layout()
    plt.show()
    
    print("\n🎉 CONGRATULATIONS!")
    print("You just learned how gradient descent works mathematically!")
    print("\nKey takeaways:")
    print("✅ Predictions = m×x + b")
    print("✅ Errors = prediction - actual")
    print("✅ Cost = average of squared errors")
    print("✅ Gradients tell us which direction to move parameters")
    print("✅ We update parameters to reduce cost")
    print("✅ Repeat until cost is minimized!")

if __name__ == "__main__":
    interactive_math_builder()
