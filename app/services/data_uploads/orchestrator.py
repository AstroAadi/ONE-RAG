from pathlib import Path
from app.services.extraction.extractor_factory import ExtractorFactory
from app.utils.markdown_converter import MarkdownConverter


# Ensure uploads folder exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def run_extraction_pipeline(folder_path: str):
    """
    Orchestrator:
    - Takes normalized folder path
    - Runs all extractors (ZIP, PDF, etc.)
    - Combines output into single markdown
    - Saves into uploads/output.md
    - Returns final markdown content for downstream chunking/embedding
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise Exception(f"Folder does not exist: {folder_path}")

    factory = ExtractorFactory()

    all_markdown = []
    files_to_process = [folder] if folder.is_file() else sorted(
        path for path in folder.rglob("*") if path.is_file()
    )

    print(f"Starting extraction pipeline on: {folder_path}")

    # Loop through all files recursively
    for file in files_to_process:
        print(f"Processing: {file.name}")

        try:
            # Get appropriate extractor
            extractor = factory.get_extractor(str(file))

            if not extractor:
                format_status = factory.get_format_status(file.suffix)
                if format_status["status"] == "unavailable":
                    print(
                        f"Skipping ({file.suffix} unavailable): {file.name} - "
                        f"{format_status['reason']}"
                    )
                else:
                    print(f"Skipping (no extractor): {file.name}")
                continue

            # Run extraction
            markdown, success, metadata = extractor.extract(str(file))

            if success and markdown.strip():
                all_markdown.append(markdown)
                print(f"Extracted: {file.name}")
            else:
                print(f"Failed extraction: {file.name}")

        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")

    # Combine all markdown
    final_markdown = "\n\n".join(all_markdown)
    final_markdown = MarkdownConverter.convert_text_to_markdown(
        final_markdown,
        folder.name,
        folder.suffix.lstrip('.') if folder.is_file() else "folder"
    )

    # Save final output
    output_file = UPLOAD_DIR / "output.md"

    if not MarkdownConverter.save_markdown_file(final_markdown, str(output_file)):
        raise Exception(f"Failed to save markdown output: {output_file}")

    print("Extraction pipeline completed")
    print(f"Output saved at: {output_file.resolve()}")

    return final_markdown
