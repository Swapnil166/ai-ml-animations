"""
Gradient Descent Learning Hub - Interactive Python Animations
Main script to explore gradient descent and linear regression concepts
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
    print("\n" + "="*70)
    print("📈 GRADIENT DESCENT LEARNING HUB - Python Animations")
    print("="*70)
    print("Choose a concept to explore:")
    print()
    print("🌟 BEGINNER FRIENDLY:")
    print("1. 🎨 Visual Math Tutorial")
    print("   Learn gradient descent with hill-climbing analogies")
    print()
    print("2. 🧮 Math Builder")
    print("   Build mathematical understanding step by step")
    print()
    print("🎓 COMPREHENSIVE LEARNING:")
    print("3. 📚 Complete Gradient Descent Tutorial")
    print("   In-depth explanation with theory and practice")
    print()
    print("4. 🎬 Complete Visualization")
    print("   All-in-one animated demonstration")
    print()
    print("⚡ QUICK DEMOS:")
    print("5. 🚀 Basic Animation")
    print("   Simple gradient descent animation")
    print()
    print("🚀 BATCH OPERATIONS:")
    print("6. 📚 Run All Demos")
    print("   Experience all animations in sequence")
    print()
    print("7. ❓ Help & Concepts")
    print("   Learn about gradient descent concepts")
    print()
    print("0. 🚪 Exit")
    print("="*70)

def show_help():
    """Display help information"""
    print("\n" + "="*70)
    print("📚 GRADIENT DESCENT CONCEPTS EXPLAINED")
    print("="*70)
    print()
    print("📈 LINEAR REGRESSION:")
    print("   - Finding the best line through data points")
    print("   - Minimizing prediction errors")
    print("   - Example: Predicting house prices from size")
    print()
    print("⛰️  GRADIENT DESCENT:")
    print("   - Algorithm to find optimal parameters")
    print("   - Like rolling a ball downhill to find the bottom")
    print("   - Uses derivatives to find the steepest descent")
    print()
    print("💰 COST FUNCTION:")
    print("   - Measures how wrong our predictions are")
    print("   - Lower cost = better predictions")
    print("   - Usually Mean Squared Error (MSE)")
    print()
    print("🎯 LEARNING RATE:")
    print("   - Controls how big steps we take")
    print("   - Too high: might overshoot the minimum")
    print("   - Too low: takes forever to converge")
    print()
    print("🔄 CONVERGENCE:")
    print("   - When the algorithm stops improving")
    print("   - Cost function reaches minimum")
    print("   - Parameters stabilize")
    print()
    print("💡 LEARNING TIPS:")
    print("   - Start with visual analogies (Demo 1)")
    print("   - Build math understanding gradually (Demo 2)")
    print("   - See complete picture (Demo 3-4)")
    print("   - Experiment with different parameters")
    print()
    print("🔗 CONCEPT CONNECTIONS:")
    print("   Visual Intuition → Mathematical Foundation → Implementation")
    print("="*70)

def run_demo(demo_number):
    """Run a specific demo"""
    demos = {
        1: "visual_math_tutorial.py",
        2: "math_builder.py",
        3: "learn_gradient_descent.py",
        4: "complete_visualization.py",
        5: "gradient_descent_animation.py"
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
    print("\n🎬 Running all Gradient Descent animation demos...")
    print("Close each plot window to proceed to the next demo.")
    
    demos = [1, 2, 3, 4, 5]
    demo_names = [
        "Visual Math Tutorial",
        "Math Builder", 
        "Complete Tutorial",
        "Complete Visualization",
        "Basic Animation"
    ]
    
    for i, name in zip(demos, demo_names):
        input(f"\nPress Enter to start: {name}...")
        run_demo(i)
    
    print("\n🎉 All gradient descent demos completed! You've learned:")
    print("   ✅ Visual intuition with hill-climbing analogies")
    print("   ✅ Mathematical foundations step by step")
    print("   ✅ Complete theoretical understanding")
    print("   ✅ Comprehensive animated visualization")
    print("   ✅ Basic implementation concepts")

def main():
    """Main application loop"""
    print("🌟 Welcome to Gradient Descent Learning with Python Animations!")
    print("🎯 Master optimization algorithms through visual learning!")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    while True:
        display_menu()
        
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
            
            if choice == '0':
                print("\n👋 Thanks for learning Gradient Descent! Happy optimizing!")
                break
            elif choice in ['1', '2', '3', '4', '5']:
                run_demo(int(choice))
            elif choice == '6':
                run_all_demos()
            elif choice == '7':
                show_help()
            else:
                print("❌ Invalid choice! Please enter a number between 0-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
