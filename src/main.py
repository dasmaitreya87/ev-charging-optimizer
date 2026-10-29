import os
import sys
import subprocess
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_script(script_path, step_name):
    """Run a Python script and handle errors"""
    print_header(f"STEP: {step_name}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed successfully in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {step_name}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_path}")
        return False

def check_requirements():
    """Check if all required packages are installed"""
    print_header("CHECKING REQUIREMENTS")
    
    required_packages = [
        'osmnx',
        'networkx',
        'pandas',
        'numpy',
        'matplotlib',
        'folium',
        'geopandas',
        'shapely',
        'requests',
        'seaborn'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install them with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✓ All required packages installed")
        return True

def create_directories():
    """Create necessary directories"""
    print_header("CREATING DIRECTORIES")
    
    dirs = [
        'data/raw',
        'data/processed',
        'results',
        'notebooks',
        'src'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ {dir_path}")
    
    print("\n✓ All directories created")

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Step 0: Check requirements
    if not check_requirements():
        print("\n⚠ Please install missing packages before continuing")
        print("Run: pip install -r requirements.txt")
        return
    
    # Step 1: Create directories
    create_directories()
     
    # Define pipeline
    # Note: Uses Google Maps API for real traffic data (requires API key)
    import os
    
    # Check if Google Maps API key is available
    # Try to import from google_traffic_api module first
    try:
        import sys
        sys.path.insert(0, 'src')
        from google_traffic_api import GOOGLE_MAPS_API_KEY
        has_api_key = GOOGLE_MAPS_API_KEY is not None and GOOGLE_MAPS_API_KEY != ""
    except:
        has_api_key = os.getenv('GOOGLE_MAPS_API_KEY') is not None
    
    if has_api_key:
        print("\n✓ Google Maps API key detected - will use real traffic data")
    else:
        print("\n⚠ Google Maps API key not found")
        print("  Set GOOGLE_MAPS_API_KEY variable in src/google_traffic_api.py to use real traffic data")
        print("  Get API key from: https://console.cloud.google.com/google/maps-apis")
    
    pipeline = [
        ('src/download_network.py', 'Download Road Network'),
        ('src/charging_stations.py', 'Download Charging Stations'),
        ('src/google_traffic_api.py', 'Generate Real Traffic Data (Google Maps API)')
    ]
    
    print("\n" + "="*70)
    print("NOTE: After running the pipeline, use the web server for route optimization:")
    print("  Run: python run_web_server.py")
    print("  Then open: http://127.0.0.1:5000")
    print("="*70)
    
    # Ask user for confirmation
    print("\n" + "="*70)
    print("PIPELINE STEPS:")
    for i, (script, name) in enumerate(pipeline, 1):
        print(f"  {i}. {name}")
    print("="*70)
    
    # Execute pipeline
    start_time = time.time()
    
    for script_path, step_name in pipeline:
        success = run_script(script_path, step_name)
        
        if not success:
            print(f"\n⚠ Pipeline stopped due to error in: {step_name}")
            print("Fix the error and run again, or run remaining scripts manually")
            return
        
        # Small pause between steps
        time.sleep(1)
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")
    print("\nGenerated outputs:")
    print("  📁 data/raw/ - Original network and data")
    print("  📁 data/processed/ - Processed datasets")
    print("  📁 results/ - Maps, plots, and analysis")
    print("\nNext steps:")
    print("  1. Check results/experiment_results.csv for detailed results")
    print("  2. Open results/best_route_map.html in your browser")
    print("  3. Review results/*.png for visualizations")
    print("  4. Use the data for your project report")
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        print("Progress has been saved. You can resume by running individual scripts.")