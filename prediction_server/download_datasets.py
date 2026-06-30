import os
import urllib.request
import zipfile
import argparse

# URLs for sample datasets
# Backblaze provides daily CSVs in zipped folders by quarter. We download a small recent quarter.
BACKBLAZE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2023.zip"

# Alibaba dcbrain provides SSD data. We use a sample URL from their open data.
ALIBABA_URL = "https://github.com/alibaba-edu/dcbrain/raw/master/ssd_smart_logs/sample_data.csv"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print("Download complete.")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    parser = argparse.ArgumentParser(description="Download sample drive failure datasets.")
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory to store datasets')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    hdd_dir = os.path.join(args.data_dir, 'hdd')
    ssd_dir = os.path.join(args.data_dir, 'ssd')
    os.makedirs(hdd_dir, exist_ok=True)
    os.makedirs(ssd_dir, exist_ok=True)

    # Download HDD Data (Backblaze) - Full Year 2023
    quarters = ["Q1_2023", "Q2_2023", "Q3_2023", "Q4_2023"]
    for q in quarters:
        url = f"https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_{q}.zip"
        zip_path = os.path.join(hdd_dir, f"data_{q}.zip")
        if not os.path.exists(zip_path):
            try:
                download_file(url, zip_path)
                extract_zip(zip_path, hdd_dir)
            except Exception as e:
                print(f"Failed to download Backblaze {q} data: {e}")
        else:
            print(f"Backblaze {q} data already downloaded.")

    # Download SSD Data (Alibaba)
    # Alibaba's repo has many files, we fetch a single sample CSV if available or create a mock sample to ensure the script doesn't crash if the exact raw URL changes.
    alibaba_file = os.path.join(ssd_dir, "alibaba_ssd_sample.csv")
    if not os.path.exists(alibaba_file):
        try:
            download_file(ALIBABA_URL, alibaba_file)
        except Exception as e:
            print(f"Failed to download Alibaba data from raw URL, creating a fallback sample. Error: {e}")
            # Fallback to create a tiny mock CSV so the pipeline can still be tested
            with open(alibaba_file, "w") as f:
                f.write("serial_number,date,smart_5,smart_173,smart_187,smart_194,smart_241,smart_242,failure\n")
                f.write("SSD_1,2023-01-01,0,100,0,30,1000,500,0\n")
                f.write("SSD_1,2023-01-15,0,95,0,31,2000,1000,0\n")
                f.write("SSD_2,2023-01-01,10,50,5,45,50000,20000,0\n")
                f.write("SSD_2,2023-01-15,50,10,20,55,100000,50000,1\n")
            print("Fallback sample created for SSD.")
    else:
        print("Alibaba data already downloaded.")

    print(f"\nDatasets are ready in {args.data_dir}. You can now run train.py.")

if __name__ == "__main__":
    main()
