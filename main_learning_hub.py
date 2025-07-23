"""
AI/ML Interactive Learning Animations - Master Hub
Main entry point for all machine learning educational modules
"""

import sys
import os
import subprocess

def check_global_dependencies():
    """Check if basic packages are installed"""
    required_packages = ['matplotlib', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nTo install all dependencies, run:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def display_main_menu():
    """Display the master menu"""
    print("\n" + "="*80)
    print("🤖 AI/ML INTERACTIVE LEARNING ANIMATIONS - MASTER HUB")
    print("="*80)
    print("🎯 Choose a learning module to explore:")
    print()
    print("🧠 NATURAL LANGUAGE PROCESSING:")
    print("1. 🔤 NLP Learning Hub")
    print("   Text processing, sentiment analysis, transformers & attention")
    print("   📊 7 interactive animations | 🎯 Beginner to Advanced")
    print()
    print("📈 MACHINE LEARNING FUNDAMENTALS:")
    print("2. ⛰️  Gradient Descent Hub")
    print("   Linear regression, optimization, cost functions")
    print("   📊 5 interactive animations | 🎯 Visual to Mathematical")
    print()
    print("3. 📊 Multiple Linear Regression Hub")
    print("   Multi-feature predictions, matrix operations")
    print("   📊 4 interactive animations | 🎯 Simple to Complex")
    print()
    print("🚀 QUICK ACCESS:")
    print("4. 🎬 Run Featured Demos")
    print("   Best animations from each module")
    print()
    print("5. 📚 Complete Learning Journey")
    print("   Guided path through all modules")
    print()
    print("6. 📖 Learning Paths & Help")
    print("   Recommended sequences and concept explanations")
    print()
    print("7. 📊 Repository Statistics")
    print("   Project overview and module details")
    print()
    print("0. 🚪 Exit")
    print("="*80)

def show_learning_paths():
    """Display recommended learning paths"""
    print("\n" + "="*80)
    print("📚 RECOMMENDED LEARNING PATHS")
    print("="*80)
    print()
    print("🌟 COMPLETE BEGINNER PATH (2-3 weeks):")
    print("   Week 1: Gradient Descent Hub → Visual Math Tutorial")
    print("           Multiple Regression Hub → Simple Introduction")
    print("   Week 2: NLP Hub → Demos 1-3 (Tokenization, Frequency, Sentiment)")
    print("           NLP Hub → Demos 4-5 (Text Preprocessing)")
    print("   Week 3: NLP Hub → Demos 6-7 (Attention Mechanisms)")
    print("           Review and practice with your own data")
    print()
    print("🧮 MATH-FOCUSED PATH (1-2 weeks):")
    print("   Day 1-2: Gradient Descent Hub → Math Builder")
    print("   Day 3-4: Multiple Regression Hub → Theory and Math")
    print("   Day 5-7: NLP Hub → Advanced Preprocessing & Attention")
    print("   Week 2: Implementation and experimentation")
    print()
    print("🎬 VISUAL LEARNER PATH (1-2 weeks):")
    print("   Start: All 'Complete Visualization' and 'Animated' demos")
    print("   Then: Interactive exploration with parameter changes")
    print("   Finally: Read code and understand implementation")
    print()
    print("🎯 SKILL-SPECIFIC PATHS:")
    print("   📊 Data Science Focus: Gradient Descent → Multiple Regression")
    print("   🧠 NLP Focus: All NLP modules → Text preprocessing → Transformers")
    print("   🤖 AI Research Focus: All modules → Advanced concepts → Implementation")
    print()
    print("💡 LEARNING TIPS:")
    print("   • Start with visual intuition before mathematical details")
    print("   • Practice with your own datasets after each module")
    print("   • Join online communities to discuss concepts")
    print("   • Build projects combining multiple techniques")
    print("="*80)

def show_repository_stats():
    """Display repository statistics and overview"""
    print("\n" + "="*80)
    print("📊 REPOSITORY STATISTICS & OVERVIEW")
    print("="*80)
    
    # Count files in each directory
    modules = {
        'NLP': 'Natural Language Processing',
        'gradient_descent': 'Gradient Descent & Linear Regression', 
        'multiple_linear_regression': 'Multiple Linear Regression'
    }
    
    total_scripts = 0
    total_animations = 0
    
    for module_dir, module_name in modules.items():
        if os.path.exists(module_dir):
            py_files = [f for f in os.listdir(module_dir) if f.endswith('.py')]
            script_count = len(py_files)
            total_scripts += script_count
            
            print(f"\n📂 {module_name}:")
            print(f"   📄 Python scripts: {script_count}")
            print(f"   🎬 Interactive animations: {script_count - 1}")  # Excluding hub file
            total_animations += (script_count - 1)
            
            # Show key files
            hub_file = next((f for f in py_files if 'hub' in f.lower()), None)
            if hub_file:
                print(f"   🎯 Learning hub: {hub_file}")
    
    print(f"\n🎯 TOTAL PROJECT STATISTICS:")
    print(f"   📄 Total Python scripts: {total_scripts}")
    print(f"   🎬 Total interactive animations: {total_animations}")
    print(f"   📂 Learning modules: {len(modules)}")
    print(f"   🎓 Learning hubs: {len(modules)}")
    
    print(f"\n🌟 KEY FEATURES:")
    print(f"   ✅ Progressive difficulty levels")
    print(f"   ✅ Visual-first learning approach")
    print(f"   ✅ Interactive parameter controls")
    print(f"   ✅ Real-world examples and applications")
    print(f"   ✅ Comprehensive documentation")
    print(f"   ✅ Modular, extensible design")
    
    print(f"\n🔗 REPOSITORY:")
    print(f"   🌐 GitHub: https://github.com/Swapnil166/ai-ml-animations")
    print(f"   📋 License: MIT (Open Source)")
    print(f"   🤝 Contributions: Welcome!")
    print("="*80)

def run_module_hub(module_name):
    """Run a specific module's learning hub"""
    module_paths = {
        'nlp': 'NLP/nlp_learning_hub.py',
        'gradient': 'gradient_descent/gradient_descent_hub.py',
        'regression': 'multiple_linear_regression/regression_hub.py'
    }
    
    if module_name in module_paths:
        script_path = module_paths[module_name]
        
        if not os.path.exists(script_path):
            print(f"❌ Module hub not found: {script_path}")
            return
        
        print(f"\n🚀 Launching {script_path}...")
        print("Follow the module's menu system to explore animations.")
        
        try:
            # Change to the module directory and run the hub
            module_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)
            
            subprocess.run([sys.executable, script_name], 
                         cwd=module_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running module hub: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    else:
        print("❌ Invalid module name!")

def run_featured_demos():
    """Run the best animations from each module"""
    print("\n🎬 Running Featured Demos from Each Module...")
    print("These are the most impressive animations from each learning area.")
    print()
    
    featured_demos = [
        ("NLP", "NLP/nlp_learning_hub.py", "NLP Learning Hub - Complete Experience"),
        ("Gradient Descent", "gradient_descent/complete_visualization.py", "All-in-One ML Visualization"),
        ("Multiple Regression", "multiple_linear_regression/animated_visualization.py", "Real-time Learning Animation")
    ]
    
    for module, script_path, description in featured_demos:
        if os.path.exists(script_path):
            print(f"\n📊 {module}: {description}")
            choice = input("Run this demo? (y/n): ").strip().lower()
            
            if choice == 'y':
                try:
                    module_dir = os.path.dirname(script_path)
                    script_name = os.path.basename(script_path)
                    subprocess.run([sys.executable, script_name], 
                                 cwd=module_dir, check=True)
                except Exception as e:
                    print(f"❌ Error running {script_path}: {e}")
        else:
            print(f"⚠️  {script_path} not found, skipping...")
    
    print("\n🎉 Featured demos completed!")

def run_complete_journey():
    """Guide user through complete learning journey"""
    print("\n🎓 Welcome to the Complete AI/ML Learning Journey!")
    print("This guided experience will take you through all concepts systematically.")
    print()
    
    journey_steps = [
        ("Step 1: Visual Foundation", "gradient_descent", "Start with visual intuition"),
        ("Step 2: Mathematical Understanding", "gradient", "Build mathematical foundation"),
        ("Step 3: Multiple Features", "regression", "Extend to complex problems"),
        ("Step 4: Text Processing", "nlp", "Enter the world of NLP"),
        ("Step 5: Advanced AI", "nlp", "Master transformer attention")
    ]
    
    for step_name, module, description in journey_steps:
        print(f"\n📚 {step_name}: {description}")
        choice = input("Continue to this step? (y/n/q to quit): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'y':
            run_module_hub(module)
            input("\nPress Enter when ready for the next step...")
        
    print("\n🎉 Congratulations on completing your AI/ML learning journey!")

def main():
    """Main application loop"""
    print("🌟 Welcome to AI/ML Interactive Learning Animations!")
    print("🎯 Your comprehensive resource for visual machine learning education")
    
    # Check dependencies
    if not check_global_dependencies():
        return
    
    while True:
        display_main_menu()
        
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
            
            if choice == '0':
                print("\n👋 Thanks for exploring AI/ML with us!")
                print("🚀 Keep learning, keep growing, keep innovating!")
                break
            elif choice == '1':
                run_module_hub('nlp')
            elif choice == '2':
                run_module_hub('gradient')
            elif choice == '3':
                run_module_hub('regression')
            elif choice == '4':
                run_featured_demos()
            elif choice == '5':
                run_complete_journey()
            elif choice == '6':
                show_learning_paths()
            elif choice == '7':
                show_repository_stats()
            else:
                print("❌ Invalid choice! Please enter a number between 0-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
