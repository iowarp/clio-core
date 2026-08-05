import subprocess
import sys
import os
import time

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"🚀 Step: {description}")
    print(f"{'='*50}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during: {description}")
        print(e)
        sys.exit(1)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Install dependencies
    run_command(f"pip install -r {os.path.join(base_dir, 'requirements.txt')}", "Installing Python dependencies")
    
    # 2. Download Datasets
    run_command(f"python {os.path.join(base_dir, 'download_datasets.py')} --data-dir {data_dir}", "Downloading Backblaze (HDD) and Alibaba (SSD) datasets")
    
    # 3. Train HDD Models (limit to 5 files to prevent OOM)
    run_command(f"python {os.path.join(base_dir, 'train.py')} --data-dir {os.path.join(data_dir, 'hdd', 'data_Q1_2023')} --drive-type hdd --limit-files 5", "Training HDD Models")
    
    # 4. Train SSD Models
    run_command(f"python {os.path.join(base_dir, 'train.py')} --data-dir {os.path.join(data_dir, 'ssd')} --drive-type ssd", "Training SSD Models")
    
    # 5. Start the API server
    print(f"\n{'='*50}")
    print(f"🚀 Step: Starting Prediction API Server")
    print(f"{'='*50}")
    print("The server will start on http://0.0.0.0:8000")
    print("Press Ctrl+C to stop.")
    
    try:
        subprocess.run(f"python {os.path.join(base_dir, 'server.py')}", shell=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
