"""
NLP Learning Hub - Interactive Python Animations
Main script to explore different NLP concepts through animations
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['matplotlib', 'numpy', 'textblob']
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
        print("\nTo install missing packages, run:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def display_menu():
    """Display the main menu"""
    print("\n" + "="*75)
    print("🧠 NLP LEARNING HUB - Python Animations")
    print("="*75)
    print("Choose a concept to explore:")
    print()
    print("📚 BASIC NLP CONCEPTS:")
    print("1. 🔤 Tokenization Animation")
    print("   Learn how text is broken down into tokens")
    print()
    print("2. 📊 Word Frequency Animation") 
    print("   See how word frequencies build up over time")
    print()
    print("3. 🎭 Sentiment Analysis Animation")
    print("   Visualize sentiment changes across different texts")
    print()
    print("🔧 TEXT PREPROCESSING:")
    print("4. 🧹 Text Preprocessing Pipeline")
    print("   Complete preprocessing steps with detailed animations")
    print()
    print("5. 🔬 Advanced Preprocessing Techniques")
    print("   N-grams, TF-IDF, and normalization methods")
    print()
    print("🤖 TRANSFORMER CONCEPTS:")
    print("6. 🎯 Attention Mechanism Animation")
    print("   Understand how attention works in transformers")
    print()
    print("7. 🔄 Multi-Head Attention Animation")
    print("   See how multiple attention heads work together")
    print()
    print("🚀 BATCH OPERATIONS:")
    print("8. 📚 Run Basic NLP Demos (1-3)")
    print("   Experience basic NLP animations in sequence")
    print()
    print("9. 🔧 Run Preprocessing Demos (4-5)")
    print("   Experience preprocessing animations in sequence")
    print()
    print("10. 🤖 Run Transformer Demos (6-7)")
    print("    Experience transformer animations in sequence")
    print()
    print("11. 🌟 Run All Demos")
    print("    Experience all animations in sequence")
    print()
    print("12. ❓ Help & Tips")
    print("    Learn more about NLP concepts")
    print()
    print("0. 🚪 Exit")
    print("="*75)

def show_help():
    """Display help information"""
    print("\n" + "="*75)
    print("📚 NLP CONCEPTS EXPLAINED")
    print("="*75)
    print()
    print("🔤 TOKENIZATION:")
    print("   - Breaking text into individual words or tokens")
    print("   - Foundation of most NLP tasks")
    print("   - Example: 'Hello world' → ['Hello', 'world']")
    print()
    print("📊 WORD FREQUENCY:")
    print("   - Counting how often words appear in text")
    print("   - Helps identify important terms")
    print("   - Used in search engines and text analysis")
    print()
    print("🎭 SENTIMENT ANALYSIS:")
    print("   - Determining emotional tone of text")
    print("   - Positive, negative, or neutral classification")
    print("   - Used in social media monitoring, reviews")
    print()
    print("🧹 TEXT PREPROCESSING:")
    print("   - Cleaning and standardizing raw text")
    print("   - Lowercase, remove punctuation, stop words")
    print("   - Essential for consistent model input")
    print("   - Improves model performance significantly")
    print()
    print("🔬 ADVANCED PREPROCESSING:")
    print("   - N-grams: sequences of consecutive words")
    print("   - TF-IDF: term importance across documents")
    print("   - Text normalization: standardizing variations")
    print("   - Feature extraction for machine learning")
    print()
    print("🎯 ATTENTION MECHANISM:")
    print("   - Allows models to focus on relevant parts of input")
    print("   - Uses Query, Key, Value vectors")
    print("   - Foundation of transformer models (GPT, BERT)")
    print("   - Enables handling of long sequences")
    print()
    print("🔄 MULTI-HEAD ATTENTION:")
    print("   - Multiple attention mechanisms working in parallel")
    print("   - Different heads focus on different patterns")
    print("   - Syntactic, semantic, positional relationships")
    print("   - Provides richer representations")
    print()
    print("💡 LEARNING TIPS:")
    print("   - Start with basic concepts (1-3) before advanced")
    print("   - Master preprocessing (4-5) before transformers (6-7)")
    print("   - Watch animations carefully to understand processes")
    print("   - Try modifying the code with your own text")
    print("   - Experiment with different parameters")
    print("   - Read the code to understand implementation")
    print()
    print("🔗 CONCEPT CONNECTIONS:")
    print("   Tokenization → Word Frequency → Sentiment Analysis")
    print("   ↓")
    print("   Text Preprocessing → Advanced Preprocessing")
    print("   ↓")
    print("   Attention Mechanism → Multi-Head Attention → Transformers")
    print("="*75)

def run_demo(demo_number):
    """Run a specific demo"""
    demos = {
        1: "01_tokenization_animation.py",
        2: "02_word_frequency_animation.py", 
        3: "03_sentiment_animation.py",
        4: "06_text_preprocessing_animation.py",
        5: "07_advanced_preprocessing_animation.py",
        6: "04_attention_mechanism_animation.py",
        7: "05_multihead_attention_animation.py"
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

def run_basic_demos():
    """Run basic NLP demos (1-3)"""
    print("\n🎬 Running Basic NLP animation demos...")
    print("Close each plot window to proceed to the next demo.")
    
    basic_demos = [1, 2, 3]
    for i in basic_demos:
        input(f"\nPress Enter to start Demo {i}...")
        run_demo(i)

def run_preprocessing_demos():
    """Run preprocessing demos (4-5)"""
    print("\n🔧 Running Text Preprocessing animation demos...")
    print("Close each plot window to proceed to the next demo.")
    
    preprocessing_demos = [4, 5]
    for i in preprocessing_demos:
        input(f"\nPress Enter to start Demo {i}...")
        run_demo(i)

def run_transformer_demos():
    """Run transformer demos (6-7)"""
    print("\n🤖 Running Transformer animation demos...")
    print("Close each plot window to proceed to the next demo.")
    
    transformer_demos = [6, 7]
    for i in transformer_demos:
        input(f"\nPress Enter to start Demo {i}...")
        run_demo(i)

def run_all_demos():
    """Run all demos in sequence"""
    print("\n🌟 Running all NLP animation demos...")
    print("Close each plot window to proceed to the next demo.")
    
    print("\n📚 Starting with Basic NLP Concepts...")
    run_basic_demos()
    
    print("\n🔧 Moving to Text Preprocessing...")
    run_preprocessing_demos()
    
    print("\n🤖 Finally, Advanced Transformer Concepts...")
    run_transformer_demos()
    
    print("\n🎉 All demos completed! You've learned:")
    print("   ✅ Tokenization")
    print("   ✅ Word Frequency Analysis") 
    print("   ✅ Sentiment Analysis")
    print("   ✅ Text Preprocessing Pipeline")
    print("   ✅ Advanced Preprocessing Techniques")
    print("   ✅ Attention Mechanisms")
    print("   ✅ Multi-Head Attention")

def main():
    """Main application loop"""
    print("🌟 Welcome to NLP Learning with Python Animations!")
    print("🎯 Now featuring Complete Text Preprocessing Pipeline!")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    while True:
        display_menu()
        
        try:
            choice = input("\nEnter your choice (0-12): ").strip()
            
            if choice == '0':
                print("\n👋 Thanks for learning NLP! Happy coding!")
                break
            elif choice in ['1', '2', '3', '4', '5', '6', '7']:
                run_demo(int(choice))
            elif choice == '8':
                run_basic_demos()
            elif choice == '9':
                run_preprocessing_demos()
            elif choice == '10':
                run_transformer_demos()
            elif choice == '11':
                run_all_demos()
            elif choice == '12':
                show_help()
            else:
                print("❌ Invalid choice! Please enter a number between 0-12.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
