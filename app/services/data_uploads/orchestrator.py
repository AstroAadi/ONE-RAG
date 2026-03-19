from pathlib import Path
from app.services.extraction.extractor_factory import ExtractorFactory


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
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise Exception(f"Folder does not exist: {folder_path}")

    factory = ExtractorFactory()

    all_markdown = []

    print(f"🚀 Starting extraction pipeline on: {folder_path}")

    # Loop through all files recursively
    for file in folder.rglob("*"):

        if not file.is_file():
            continue

        print(f"📄 Processing: {file.name}")

        try:
            # Get appropriate extractor
            extractor = factory.get_extractor(str(file))

            if not extractor:
                print(f"⏭ Skipping (no extractor): {file.name}")
                continue

            # Run extraction
            markdown, success, metadata = extractor.extract(str(file))

            if success and markdown.strip():
                all_markdown.append(markdown)
                print(f"✅ Extracted: {file.name}")
            else:
                print(f"⚠️ Failed extraction: {file.name}")

        except Exception as e:
            print(f"❌ Error processing {file.name}: {str(e)}")

    # Combine all markdown
    final_markdown = "\n\n".join(all_markdown)

    # Save final output
    output_file = UPLOAD_DIR / "output.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    print("🎉 Extraction pipeline completed")
    print(f"📁 Output saved at: {output_file.resolve()}")

    return str(output_file)