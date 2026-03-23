import logging
import time

import boto3
from datetime import datetime
from pathlib import Path, PurePosixPath

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
# MAIN FUNCTION (UNCHANGED LOGIC)
# ==============================
def process_s3_data(access_key, secret_key, bucket_name, s3_prefix):
    start_time = time.perf_counter()
    download_dir = _create_download_dir("aws")
    logger.info(
        "AWS download started: bucket=%s prefix=%s download_dir=%s",
        bucket_name,
        s3_prefix or "/",
        download_dir.resolve(),
    )

    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    logger.info("AWS S3 client created successfully")

    logger.info("Listing S3 objects for bucket=%s prefix=%s", bucket_name, s3_prefix or "/")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
    downloaded_files = []
    total_bytes = 0
    objects = response.get('Contents', [])
    logger.info(
        "S3 listing complete: %s object(s) returned (truncated=%s)",
        len(objects),
        response.get("IsTruncated", False),
    )
    if response.get("IsTruncated"):
        logger.warning(
            "S3 listing was truncated at 1000 objects for bucket=%s prefix=%s. "
            "Paginator support is not enabled yet.",
            bucket_name,
            s3_prefix or "/",
        )

    for index, obj in enumerate(objects, start=1):
        key = obj['Key']
        if key.endswith('/'):
            logger.debug("Skipping S3 folder marker %s", key)
            continue

        object_size = int(obj.get("Size", 0) or 0)
        relative_path = _build_relative_path(s3_prefix, key)
        local_path = download_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        unique_local_path = _ensure_unique_path(local_path)
        if unique_local_path != local_path:
            logger.warning(
                "Resolved S3 name collision for key=%s: %s -> %s",
                key,
                local_path,
                unique_local_path,
            )
        local_path = unique_local_path

        logger.info(
            "Downloading S3 object %s/%s: key=%s size=%s local_path=%s",
            index,
            len(objects),
            key,
            _format_bytes(object_size),
            local_path,
        )
        file_start = time.perf_counter()
        s3.download_file(bucket_name, key, str(local_path))
        elapsed = time.perf_counter() - file_start
        downloaded_files.append(local_path)
        total_bytes += object_size
        logger.info(
            "Completed S3 download %s/%s in %.2fs: %s",
            index,
            len(objects),
            elapsed,
            local_path,
        )

    if not downloaded_files:
        logger.warning(
            "AWS download finished with no files: bucket=%s prefix=%s",
            bucket_name,
            s3_prefix or "/",
        )
        raise ValueError("No files found in the provided S3 bucket/prefix")

    total_elapsed = time.perf_counter() - start_time
    logger.info(
        "AWS download complete: files=%s total_size=%s elapsed=%.2fs download_dir=%s",
        len(downloaded_files),
        _format_bytes(total_bytes),
        total_elapsed,
        download_dir.resolve(),
    )
    logger.info("ZIP files are preserved and will be handled by ZIPExtractor")

    return str(download_dir)


# ==============================
# WRAPPER FUNCTION (IMPORTANT)
# ==============================
def aws_download_handler(data: dict):
    """
    This is what your API will call
    """
    logger.info(
        "AWS upload request received: bucket=%s prefix=%s",
        data.get("bucket_name"),
        data.get("s3_prefix") or "/",
    )

    return process_s3_data(
        access_key=data["access_key"],
        secret_key=data["secret_key"],
        bucket_name=data["bucket_name"],
        s3_prefix=data["s3_prefix"]
    )
