"""Script to create submission zip file with code and report template."""

import os
import zipfile
import shutil
from pathlib import Path


def create_submission_zip(
    output_name: str = "submission.zip",
    include_artifacts: bool = False
):
    """
    Create a zip file with required code and report template.
    
    Args:
        output_name: Name of the output zip file
        include_artifacts: Whether to include artifacts directory
    """
    # Files and directories to include
    include_patterns = [
        "src/",
        "run/",
        "scripts/",
        "report/",
        "README.md",
        "requirements.txt"
    ]
    
    if include_artifacts:
        include_patterns.append("artifacts/")
    
    # Create zip file
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pattern in include_patterns:
            path = Path(pattern)
            if path.is_file():
                zipf.write(path, path)
            elif path.is_dir():
                for root, dirs, files in os.walk(path):
                    # Skip __pycache__ and .pyc files
                    dirs[:] = [d for d in dirs if d != '__pycache__']
                    files = [f for f in files if not f.endswith('.pyc')]
                    
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path
                        zipf.write(file_path, arcname)
    
    print(f"Submission zip created: {output_name}")
    print(f"Included: {', '.join(include_patterns)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create submission zip")
    parser.add_argument("--output", type=str, default="submission.zip", help="Output zip filename")
    parser.add_argument("--include-artifacts", action="store_true", help="Include artifacts directory")
    
    args = parser.parse_args()
    create_submission_zip(args.output, args.include_artifacts)

