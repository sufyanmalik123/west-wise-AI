"""
Check if Microsoft Phi-2 model is available on your machine
"""

import os
from pathlib import Path

def check_model_availability():
    """Check if Phi-2 model is cached locally"""
    
    print("="*70)
    print("🔍 CHECKING FOR MICROSOFT PHI-2 MODEL")
    print("="*70)
    
    # Common Hugging Face cache locations
    cache_locations = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "transformers",
        Path(os.environ.get('HF_HOME', '')) / "hub" if os.environ.get('HF_HOME') else None,
        Path(os.environ.get('TRANSFORMERS_CACHE', '')) if os.environ.get('TRANSFORMERS_CACHE') else None,
    ]
    
    # Remove None values
    cache_locations = [loc for loc in cache_locations if loc is not None]
    
    print("\n📂 Checking cache locations...")
    for location in cache_locations:
        print(f"   • {location}")
    
    model_found = False
    model_path = None
    
    # Look for phi-2 model
    for cache_dir in cache_locations:
        if cache_dir.exists():
            # Search for phi-2 related directories
            for item in cache_dir.iterdir():
                if 'phi-2' in item.name.lower() or 'phi_2' in item.name.lower():
                    model_found = True
                    model_path = item
                    break
        if model_found:
            break
    
    print("\n" + "="*70)
    if model_found:
        print("✅ MODEL FOUND!")
        print("="*70)
        print(f"📍 Location: {model_path}")
        print("\n✅ The model is cached on your machine.")
        print("✅ Script will run faster (no download needed).")
    else:
        print("❌ MODEL NOT FOUND")
        print("="*70)
        print("\n⚠️  The Phi-2 model is NOT cached on your machine.")
        print("\n📥 What will happen when you run the script:")
        print("   1. Script will download the model (~5GB)")
        print("   2. Takes 5-10 minutes depending on internet speed")
        print("   3. Model will be cached for future use")
        print("   4. Subsequent runs will be much faster")
    
    print("\n" + "="*70)
    print("📋 CACHE DIRECTORY INFO")
    print("="*70)
    
    for cache_dir in cache_locations:
        if cache_dir.exists():
            print(f"\n📂 {cache_dir}")
            print(f"   Exists: ✅ Yes")
            # Check size if possible
            try:
                items = list(cache_dir.iterdir())
                print(f"   Items: {len(items)} files/folders")
            except:
                print(f"   Items: Unable to count")
        else:
            print(f"\n📂 {cache_dir}")
            print(f"   Exists: ❌ No")
    
    print("\n" + "="*70)
    return model_found


def check_dependencies():
    """Check if required libraries are installed"""
    print("\n🔧 CHECKING DEPENDENCIES")
    print("="*70)
    
    dependencies = {
        'transformers': False,
        'torch': False,
        'pandas': False
    }
    
    for package in dependencies.keys():
        try:
            __import__(package)
            dependencies[package] = True
            print(f"✅ {package}: Installed")
        except ImportError:
            dependencies[package] = False
            print(f"❌ {package}: NOT installed")
    
    print("\n" + "="*70)
    
    if all(dependencies.values()):
        print("✅ All dependencies are installed!")
        return True
    else:
        print("⚠️  Some dependencies are missing.")
        print("\n📦 To install missing packages:")
        missing = [pkg for pkg, installed in dependencies.items() if not installed]
        print(f"   pip install {' '.join(missing)}")
        return False


if __name__ == "__main__":
    print("\n🚀 MICROSOFT PHI-2 MODEL CHECKER")
    print("="*70 + "\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check model availability
    model_available = check_model_availability()
    
    # Final summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    if deps_ok and model_available:
        print("✅ Everything is ready!")
        print("✅ You can run the analyzer immediately.")
        print("\n🚀 Run: python phi2_keyword_analyzer.py")
    elif deps_ok and not model_available:
        print("⚠️  Dependencies installed, but model not cached.")
        print("📥 First run will download the model (~5GB)")
        print("\n🚀 Run: python phi2_keyword_analyzer.py")
        print("   (First run: 5-10 minutes for download)")
    elif not deps_ok and model_available:
        print("⚠️  Model found, but dependencies missing.")
        print("\n📦 Install dependencies first:")
        print("   pip install transformers torch pandas")
    else:
        print("⚠️  Setup required.")
        print("\n📦 Step 1: Install dependencies")
        print("   pip install transformers torch pandas")
        print("\n📥 Step 2: Run analyzer (will download model)")
        print("   python phi2_keyword_analyzer.py")
    
    print("="*70 + "\n")
