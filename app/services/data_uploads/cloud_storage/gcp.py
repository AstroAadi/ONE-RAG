import os
import zipfile
import shutil
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account


# ==============================
# MAIN FUNCTION (UNCHANGED)
# ==============================
def process_gcs_data(service_account_json, bucket_name, prefix):

    base_temp = Path("temp_data")
    download_dir = base_temp / "downloads"
    extracted_dir = base_temp / "extracted"
    final_dir = base_temp / "final"

    for d in [download_dir, extracted_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    credentials = service_account.Credentials.from_service_account_info(
        service_account_json
    )

    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)

    print("📥 Downloading from GCS...")

    blobs = bucket.list_blobs(prefix=prefix)

    found = False
    for blob in blobs:
        found = True

        if blob.name.endswith('/'):
            continue

        print(f"Downloading: {blob.name}")

        local_path = download_dir / os.path.basename(blob.name)
        blob.download_to_filename(str(local_path))

    if not found:
        print("❌ No files found. Check bucket/prefix.")
        return None

    print("⚙️ Processing files...")

    for file in download_dir.iterdir():

        if zipfile.is_zipfile(file):
            print(f"📦 Extracting ZIP: {file.name}")
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
        else:
            shutil.copy(file, extracted_dir / file.name)

    print("📂 Flattening structure...")

    def flatten_folder(src_folder, dest_folder):
        for root, dirs, files in os.walk(src_folder):
            for file in files:
                src_path = Path(root) / file

                dest_path = dest_folder / file
                counter = 1
                while dest_path.exists():
                    dest_path = dest_folder / f"{counter}_{file}"
                    counter += 1

                shutil.copy(src_path, dest_path)

    flatten_folder(extracted_dir, final_dir)

    print("✅ Processing Complete!")
    print(f"📁 Final Data Folder: {final_dir.resolve()}")

    return str(final_dir)


# ==============================
# WRAPPER (FOR API)
# ==============================
def gcp_download_handler(data: dict):

    # Fix private key newline issue
    data["service_account_json"]["private_key"] = data["service_account_json"]["private_key"].replace("\\n", "\n")

    return process_gcs_data(
        service_account_json=data["service_account_json"],
        bucket_name=data["bucket_name"],
        prefix=data["prefix"]
    )