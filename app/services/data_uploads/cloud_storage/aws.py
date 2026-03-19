import os
import boto3
import zipfile
import shutil
from pathlib import Path


# ==============================
# MAIN FUNCTION (UNCHANGED LOGIC)
# ==============================
def process_s3_data(access_key, secret_key, bucket_name, s3_prefix):

    base_temp = Path("temp_data")
    download_dir = base_temp / "downloads"
    extracted_dir = base_temp / "extracted"
    final_dir = base_temp / "final"

    for d in [download_dir, extracted_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    print("📥 Downloading from S3...")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)

    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith('/'):
            continue

        local_path = download_dir / os.path.basename(key)
        s3.download_file(bucket_name, key, str(local_path))

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
# WRAPPER FUNCTION (IMPORTANT)
# ==============================
def aws_download_handler(data: dict):
    """
    This is what your API will call
    """

    return process_s3_data(
        access_key=data["access_key"],
        secret_key=data["secret_key"],
        bucket_name=data["bucket_name"],
        s3_prefix=data["s3_prefix"]
    )