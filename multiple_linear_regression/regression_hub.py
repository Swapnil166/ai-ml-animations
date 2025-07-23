"""
Multiple Linear Regression Learning Hub - Interactive Python Animations
Main script to explore multiple linear regression concepts
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['matplotlib', 'numpy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nTo install missing packages, run:")
        print("pip install numpy matplotlib scikit-learn")
        return False
    
    return True

def display_menu():
    """Display the main menu"""
    print("\n" + "="*75)
    print("📊 MULTIPLE LINEAR REGRESSION LEARNING HUB - Python Animations")
    print("="*75)
    print("Choose a concept to explore:")
    print()
    print("🌱 BEGINNER FRIENDLY:")
    print("1. 🎯 Simple Introduction")
    print("   Gentle introduction to multiple features")
    print()
    print("2. ⚖️  Simple vs Multiple Comparison")
    print("   See why multiple features improve predictions")
    print()
    print("🧮 MATHEMATICAL FOUNDATION:")
    print("3. 📐 Theory and Mathematics")
    print("   Mathematical foundations and matrix operations")
    print()
    print("🎬 INTERACTIVE VISUALIZATION:")
    print("4. 🎨 Animated Visualization")
    print("   Watch multiple regression learn in real-time")
    print()
    print("🚀 BATCH OPERATIONS:")
    print("5. 📚 Run All Demos")
    print("   Experience all animations in sequence")
    print()
    print("6. ❓ Help & Concepts")
    print("   Learn about multiple regression concepts")
    print()
    print("0. 🚪 Exit")
    print("="*75)

def show_help():
    """Display help information"""
    print("\n" + "="*75)
    print("📚 MULTIPLE LINEAR REGRESSION CONCEPTS EXPLAINED")
    print("="*75)
    print()
    print("📊 MULTIPLE LINEAR REGRESSION:")
    print("   - Using multiple input features to make predictions")
    print("   - Example: House price = f(size, bedrooms, age, location)")
    print("   - More features often = better predictions")
    print()
    print("🆚 SIMPLE vs MULTIPLE:")
    print("   - Simple: y = mx + b (one feature)")
    print("   - Multiple: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b")
    print("   - Multiple captures more complex relationships")
    print()
    print("🧮 MATRIX OPERATIONS:")
    print("   - Efficient computation with many features")
    print("   - X·w = y (matrix multiplication)")
    print("   - Normal equation: w = (XᵀX)⁻¹Xᵀy")
    print()
    print("⚖️  FEATURE IMPORTANCE:")
    print("   - Different features have different impacts")
    print("   - Weight magnitude shows importance")
    print("   - Positive/negative weights show direction")
    print()
    print("📈 MODEL EVALUATION:")
    print("   - R² Score: How much variance is explained")
    print("   - Mean Squared Error: Average prediction error")
    print("   - Residual Analysis: Pattern in errors")
    print()
    print("⚠️  COMMON CHALLENGES:")
    print("   - Multicollinearity: Features too similar")
    print("   - Overfitting: Too many features")
    print("   - Feature scaling: Different units/ranges")
    print()
    print("💡 LEARNING TIPS:")
    print("   - Start simple, add complexity gradually")
    print("   - Visualize relationships between features")
    print("   - Compare simple vs multiple performance")
    print("   - Understand when to use each approach")
    print()
    print("🔗 CONCEPT CONNECTIONS:")
    print("   Simple Regression → Multiple Features → Matrix Math → Evaluation")
    print("="*75)

def run_demo(demo_number):
    """Run a specific demo"""
    demos = {
        1: "simple_introduction.py",
        2: "simple_vs_multiple.py",
        3: "theory_and_math.py",
        4: "animated_visualization.py"
    }
    
    if demo_number in demos:
        script_name = demos[demo_number]
        
        print(f"\n🚀 Running {script_name}...")
        print("Close the plot window to return to the menu.")
        
        # Check if file exists in current directory
        if not os.path.exists(script_name):
            print(f"❌ Script {script_name} not found!")
            print("Available files in current directory:")
            for file in os.listdir('.'):
                if file.endswith('.py'):
                    print(f"   - {file}")
            return
        
        try:
            subprocess.run([sys.executable, script_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {script_name}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    else:
        print("❌ Invalid demo number!")

def run_all_demos():
    """Run all demos in sequence"""
    print("\n🎬 Running all Multiple Linear Regression animation demos...")
    print("Close each plot window to proceed to the next demo.")
    
    demos = [1, 2, 3, 4]
    demo_names = [
        "Simple Introduction",
        "Simple vs Multiple Comparison",
        "Theory and Mathematics",
        "Animated Visualization"
    ]
    
    for i, name in zip(demos, demo_names):
        input(f"\nPress Enter to start: {name}...")
        run_demo(i)
    
    print("\n🎉 All multiple regression demos completed! You've learned:")
    print("   ✅ Why multiple features improve predictions")
    print("   ✅ Comparison between simple and multiple regression")
    print("   ✅ Mathematical foundations and matrix operations")
    print("   ✅ Real-time visualization of learning process")
    print("   ✅ Feature importance and model evaluation")

def main():
    """Main application loop"""
    print("🌟 Welcome to Multiple Linear Regression Learning!")
    print("📊 Master multi-feature predictions through visual learning!")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    while True:
        display_menu()
        
        try:
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == '0':
                print("\n👋 Thanks for learning Multiple Linear Regression!")
                print("🎯 Now you can handle complex, multi-feature predictions!")
                break
            elif choice in ['1', '2', '3', '4']:
                run_demo(int(choice))
            elif choice == '5':
                run_all_demos()
            elif choice == '6':
                show_help()
            else:
                print("❌ Invalid choice! Please enter a number between 0-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
