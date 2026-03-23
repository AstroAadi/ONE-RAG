import logging
import time

from datetime import datetime
from pathlib import Path, PurePosixPath
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


def _create_download_dir(source_name: str) -> Path:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    download_dir = Path("temp_data") / f"{source_name}_{run_id}" / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


def _build_relative_path(prefix: str, object_key: str) -> Path:
    normalized_key = object_key.replace("\\", "/").strip("/")
    normalized_prefix = (prefix or "").replace("\\", "/").strip("/")

    if normalized_prefix and normalized_key.startswith(f"{normalized_prefix}/"):
        relative_key = normalized_key[len(normalized_prefix) + 1:]
    else:
        relative_key = normalized_key

    relative_parts = [
        part for part in PurePosixPath(relative_key).parts
        if part not in {"", ".", ".."}
    ]

    if not relative_parts:
        relative_parts = [PurePosixPath(normalized_key).name or "downloaded_file"]

    return Path(*relative_parts)


def _ensure_unique_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    counter = 1
    while True:
        candidate = target_path.with_name(
            f"{target_path.stem}_{counter}{target_path.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def _format_bytes(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


# ==============================
# MAIN FUNCTION (UNCHANGED)
# ==============================
def process_gcs_data(service_account_json, bucket_name, prefix):
    start_time = time.perf_counter()
    download_dir = _create_download_dir("gcp")
    logger.info(
        "GCP download started: bucket=%s prefix=%s download_dir=%s",
        bucket_name,
        prefix or "/",
        download_dir.resolve(),
    )

    credentials = service_account.Credentials.from_service_account_info(
        service_account_json
    )
    logger.info("GCP credentials loaded successfully")

    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    logger.info("GCS client created successfully for bucket=%s", bucket_name)

    logger.info("Listing GCS blobs for bucket=%s prefix=%s", bucket_name, prefix or "/")
    blobs = bucket.list_blobs(prefix=prefix)

    found = False
    downloaded_files = []
    total_bytes = 0
    for index, blob in enumerate(blobs, start=1):
        found = True

        if blob.name.endswith('/'):
            logger.debug("Skipping GCS folder marker %s", blob.name)
            continue

        blob_size = int(getattr(blob, "size", 0) or 0)
        relative_path = _build_relative_path(prefix, blob.name)
        local_path = download_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        unique_local_path = _ensure_unique_path(local_path)
        if unique_local_path != local_path:
            logger.warning(
                "Resolved GCS name collision for blob=%s: %s -> %s",
                blob.name,
                local_path,
                unique_local_path,
            )
        local_path = unique_local_path

        logger.info(
            "Downloading GCS blob %s: name=%s size=%s local_path=%s",
            index,
            blob.name,
            _format_bytes(blob_size),
            local_path,
        )
        file_start = time.perf_counter()
        blob.download_to_filename(str(local_path))
        elapsed = time.perf_counter() - file_start
        downloaded_files.append(local_path)
        total_bytes += blob_size
        logger.info(
            "Completed GCS download %s in %.2fs: %s",
            index,
            elapsed,
            local_path,
        )

    if not found or not downloaded_files:
        logger.warning(
            "GCP download finished with no files: bucket=%s prefix=%s",
            bucket_name,
            prefix or "/",
        )
        return None

    total_elapsed = time.perf_counter() - start_time
    logger.info(
        "GCP download complete: files=%s total_size=%s elapsed=%.2fs download_dir=%s",
        len(downloaded_files),
        _format_bytes(total_bytes),
        total_elapsed,
        download_dir.resolve(),
    )
    logger.info("ZIP files are preserved and will be handled by ZIPExtractor")

    return str(download_dir)


# ==============================
# WRAPPER (FOR API)
# ==============================
def gcp_download_handler(data: dict):
    logger.info(
        "GCP upload request received: bucket=%s prefix=%s",
        data.get("bucket_name"),
        data.get("prefix") or "/",
    )

    # Fix private key newline issue
    data["service_account_json"]["private_key"] = data["service_account_json"]["private_key"].replace("\\n", "\n")

    return process_gcs_data(
        service_account_json=data["service_account_json"],
        bucket_name=data["bucket_name"],
        prefix=data["prefix"]
    )
