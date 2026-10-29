"""
Simple script to run the web server
Just run: python run_web_server.py
"""

import sys
import os

# Change to project root directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the server
from web_server import app, load_data

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print("="*60)
    print("EV CHARGING ROUTE OPTIMIZATION - WEB SERVER")
    print("="*60)
    
    if not load_data():
        print("\nERROR: Failed to load data. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Server starting...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
