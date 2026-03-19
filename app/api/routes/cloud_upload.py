from fastapi import APIRouter
from app.services.data_uploads.cloud_storage.aws import aws_download_handler
from app.services.data_uploads.cloud_storage.gcp import gcp_download_handler
from app.models.schemas import AWSRequest, GCPRequest
from app.services.data_uploads.orchestrator import run_extraction_pipeline

router = APIRouter()


# ✅ AWS API
@router.post("/aws-upload")
def aws_upload(data: AWSRequest):

    # 🔹 THIS IS WHERE YOUR CODE GOES
    folder_path = aws_download_handler(data.dict())

    final_md = run_extraction_pipeline(folder_path)

    return {
        "status": "success",
        "folder_path": folder_path,
        "markdown_output": final_md
    }


# ✅ GCP API
@router.post("/gcp-upload")
def gcp_upload(data: GCPRequest):

    # 🔹 SAME HERE
    folder_path = gcp_download_handler(data.dict())

    final_md = run_extraction_pipeline(folder_path)

    return {
        "status": "success",
        "folder_path": folder_path,
        "markdown_output": final_md
    }