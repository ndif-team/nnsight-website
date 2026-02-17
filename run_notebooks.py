#!/usr/bin/env python3
"""
Script to run all Jupyter notebooks using papermill.
Catches and reports any errors that occur during execution.

By default, runs notebooks in both 'features' and 'tutorials' folders.
Use --folders to specify individual folders to run.
"""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime
import papermill as pm
from papermill.exceptions import PapermillExecutionError
import nbformat

# Package versions
import nnsight
import torch
import transformers


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# Configuration
BASE_DIR = Path(__file__).parent
NOTEBOOKS_BASE = BASE_DIR / "source" / "notebooks"
OUTPUT_DIR = BASE_DIR / "notebook_outputs"
DEFAULT_FOLDERS = ["features"]

# Directories that may be created by notebook execution and should be cleaned up
ARTIFACT_DIRS = [
    "CausalAbstraction",
    "nnsight-tutorials",
]


def get_version_info() -> str:
    """Generate markdown content with package versions."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return (
        f"**Last Execution:** {timestamp}\n\n"
        f"<details>\n"
        f"<summary><b>Info</b></summary>\n\n"
        f"| Package | Version |\n"
        f"|---------|---------|\n"
        f"| **nnsight** | **{nnsight.__version__}** |\n"
        f"| Python | {python_version} |\n"
        f"| torch | {torch.__version__} |\n"
        f"| transformers | {transformers.__version__} |\n\n"
        f"</details>\n"
    )


VERSION_CELL_MARKER = "**Last Execution:**"


def get_current_versions() -> dict[str, str]:
    """Return a dict of current system package versions."""
    return {
        "nnsight": nnsight.__version__,
        "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def extract_versions_from_cell(cell_source: str) -> dict[str, str]:
    """Extract package versions from an existing version info cell."""
    import re
    versions = {}
    # Match markdown table rows: | package | version |
    for match in re.finditer(r"\|\s*\*{0,2}(\w+)\*{0,2}\s*\|\s*\*{0,2}([\w.+]+)\*{0,2}\s*\|", cell_source):
        pkg, ver = match.group(1), match.group(2)
        if pkg not in ("Package", ""):
            versions[pkg] = ver
    return versions


def has_version_mismatch(notebook_path: Path) -> bool:
    """Check if the notebook's recorded versions differ from current system versions."""
    nb = nbformat.read(notebook_path, as_version=4)

    if (
        nb.cells
        and nb.cells[0].cell_type == "markdown"
        and VERSION_CELL_MARKER in nb.cells[0].source
    ):
        recorded = extract_versions_from_cell(nb.cells[0].source)
        current = get_current_versions()
        if recorded and recorded != current:
            return True
        # No mismatch (or no recorded versions found — treat as no mismatch)
        return not recorded

    # No version cell exists yet — treat as mismatch so it gets added
    return True


def strip_papermill_metadata(nb: nbformat.NotebookNode) -> None:
    """Remove papermill metadata from notebook and cells."""
    nb.metadata.pop("papermill", None)
    for cell in nb.cells:
        cell.metadata.pop("papermill", None)


def add_version_cell(notebook_path: Path) -> None:
    """Add or update a markdown cell with version info at the beginning of a notebook."""
    nb = nbformat.read(notebook_path, as_version=4)

    strip_papermill_metadata(nb)

    # Check if the first cell is already a version cell
    if (
        nb.cells
        and nb.cells[0].cell_type == "markdown"
        and VERSION_CELL_MARKER in nb.cells[0].source
    ):
        nb.cells[0].source = get_version_info()
    else:
        version_cell = nbformat.v4.new_markdown_cell(get_version_info())
        nb.cells.insert(0, version_cell)

    nbformat.write(nb, notebook_path)


def run_notebooks(folders: list[str], skip: list[str] = None, only: list[str] = None, clean: bool = False, update: bool = False, force_update: bool = False):
    """Run all notebooks in the specified folders and collect results."""
    skip = skip or []
    only = only or []
    output_files = []  # Track output files for cleanup
    # Track mapping from output path to source path for update mode
    output_to_source: dict[Path, Path] = {}
    
    # Create output directory for executed notebooks
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Collect all notebook files from specified folders (including subfolders)
    notebooks = []
    for folder in folders:
        folder_path = NOTEBOOKS_BASE / folder
        if not folder_path.exists():
            print(f"{Colors.YELLOW}Warning: Folder '{folder}' not found at {folder_path}{Colors.RESET}")
            continue
        # Use recursive glob to find notebooks in subfolders too
        folder_notebooks = sorted(folder_path.glob("**/*.ipynb"))
        notebooks.extend(folder_notebooks)
    
    # Filter to only specified notebooks if --only is used
    if only:
        original_count = len(notebooks)
        notebooks = [
            nb for nb in notebooks
            if any(only_name in nb.stem for only_name in only)
        ]
        if notebooks:
            print(f"{Colors.CYAN}Running only {len(notebooks)} notebook(s) matching: {', '.join(only)}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}No notebooks found matching: {', '.join(only)}{Colors.RESET}")
            return 1
    
    # Filter out skipped notebooks
    skipped_notebooks = []
    if skip:
        skipped_notebooks = [
            str(nb.relative_to(NOTEBOOKS_BASE)) for nb in notebooks
            if any(skip_name in nb.stem for skip_name in skip)
        ]
        notebooks = [
            nb for nb in notebooks
            if not any(skip_name in nb.stem for skip_name in skip)
        ]
        if skipped_notebooks:
            print(f"{Colors.YELLOW}Skipping {len(skipped_notebooks)} notebook(s) matching: {', '.join(skip)}{Colors.RESET}")
    
    if not notebooks:
        print(f"{Colors.YELLOW}No notebooks found in specified folders: {folders}{Colors.RESET}")
        return 1
    
    print(f"{Colors.BOLD}Found {len(notebooks)} notebooks to run from: {', '.join(folders)}{Colors.RESET}")
    print(f"{Colors.DIM}{'=' * 60}{Colors.RESET}")
    
    results = {
        "passed": [],
        "failed": [],
        "skipped": skipped_notebooks,
        "errors": {}
    }
    
    for notebook_path in notebooks:
        notebook_name = notebook_path.name
        # Get relative path from notebooks base for proper subfolder handling
        rel_path = notebook_path.relative_to(NOTEBOOKS_BASE)
        # Use full relative path (with __ instead of /) for output filename
        output_path = OUTPUT_DIR / str(rel_path).replace("/", "__")
        display_name = str(rel_path)
        
        # Track mapping for update mode
        output_to_source[output_path] = notebook_path
        
        print(f"\n{Colors.CYAN}▶ Running: {display_name}{Colors.RESET}")
        print(f"{Colors.DIM}{'-' * 40}{Colors.RESET}")
        
        try:
            # Execute the notebook with papermill
            pm.execute_notebook(
                input_path=str(notebook_path),
                output_path=str(output_path),
                kernel_name="python3",
                progress_bar=True,
                log_output=True,
            )
            
            # Add version cell unless clean mode without any update flag
            if not clean or update or force_update:
                add_version_cell(output_path)
            output_files.append(output_path)
            print(f"{Colors.GREEN}✓ PASSED: {display_name}{Colors.RESET}")
            results["passed"].append(display_name)
            
        except PapermillExecutionError as e:
            # Still add version info to failed notebooks (unless clean mode without any update flag)
            if output_path.exists():
                if not clean or update or force_update:
                    add_version_cell(output_path)
                output_files.append(output_path)
            print(f"{Colors.RED}✗ FAILED: {display_name}{Colors.RESET}")
            print(f"{Colors.RED}  Cell {e.cell_index}: {Colors.YELLOW}{e.ename}: {e.evalue}{Colors.RESET}")
            results["failed"].append(display_name)
            results["errors"][display_name] = {
                "cell_index": e.cell_index,
                "error_name": e.ename,
                "error_value": e.evalue,
                "traceback": e.traceback,
            }
            
        except Exception as e:
            if output_path.exists():
                output_files.append(output_path)
            print(f"{Colors.RED}✗ FAILED: {display_name}{Colors.RESET}")
            print(f"{Colors.RED}  Unexpected error: {Colors.YELLOW}{type(e).__name__}: {e}{Colors.RESET}")
            results["failed"].append(display_name)
            results["errors"][display_name] = {
                "cell_index": None,
                "error_name": type(e).__name__,
                "error_value": str(e),
                "traceback": None,
            }
    
    # Print summary
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}{Colors.RESET}")
    versions = get_current_versions()
    for pkg, ver in versions.items():
        print(f"{Colors.DIM}  {pkg}: {ver}{Colors.RESET}")
    print()
    print(f"Total notebooks: {len(notebooks) + len(results['skipped'])}")
    print(f"{Colors.GREEN}Passed: {len(results['passed'])}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {len(results['failed'])}{Colors.RESET}")
    if results["skipped"]:
        print(f"{Colors.YELLOW}Skipped: {len(results['skipped'])}{Colors.RESET}")
    
    if results["passed"]:
        print(f"\n{Colors.GREEN}✓ Passed notebooks:{Colors.RESET}")
        for name in results["passed"]:
            print(f"{Colors.GREEN}  - {name}{Colors.RESET}")
    
    if results["failed"]:
        print(f"\n{Colors.RED}✗ Failed notebooks:{Colors.RESET}")
        for name in results["failed"]:
            error_info = results["errors"][name]
            cell_info = f" (cell {error_info['cell_index']})" if error_info['cell_index'] is not None else ""
            print(f"{Colors.RED}  - {name}{cell_info}{Colors.RESET}")
            print(f"{Colors.YELLOW}    {error_info['error_name']}: {error_info['error_value']}{Colors.RESET}")
    
    if results["skipped"]:
        print(f"\n{Colors.YELLOW}⊘ Skipped notebooks:{Colors.RESET}")
        for name in results["skipped"]:
            print(f"{Colors.YELLOW}  - {name}{Colors.RESET}")
    
    # Update source notebooks if requested and all passed
    updated = False
    if update or force_update:
        if results["failed"]:
            print(f"\n{Colors.RED}Cannot update: {len(results['failed'])} notebook(s) failed{Colors.RESET}")
        else:
            # Check which notebooks need updating based on version mismatch
            files_to_update = []
            for output_file in output_files:
                if output_file.exists() and output_file in output_to_source:
                    source_file = output_to_source[output_file]
                    if force_update or has_version_mismatch(source_file):
                        files_to_update.append((output_file, source_file))
                    else:
                        print(f"{Colors.DIM}  ⊘ Skipped (versions match): {source_file.relative_to(NOTEBOOKS_BASE)}{Colors.RESET}")

            if files_to_update:
                print(f"\n{Colors.GREEN}Updating {len(files_to_update)} source file(s)...{Colors.RESET}")
                for output_file, source_file in files_to_update:
                    shutil.copy2(output_file, source_file)
                    print(f"{Colors.GREEN}  ✓ Updated: {source_file.relative_to(NOTEBOOKS_BASE)}{Colors.RESET}")
                updated = True
            else:
                print(f"\n{Colors.CYAN}All notebooks already up to date (versions match){Colors.RESET}")
    
    # Clean up output files if requested (or after successful update)
    if (clean or updated) and output_files:
        for output_file in output_files:
            if output_file.exists():
                output_file.unlink()
        if clean:
            print(f"\n{Colors.DIM}Cleaned up {len(output_files)} output notebook(s){Colors.RESET}")
    
    # Clean up artifact directories created by notebook execution
    cleaned_artifacts = []
    for artifact_name in ARTIFACT_DIRS:
        artifact_path = BASE_DIR / artifact_name
        if artifact_path.exists() and artifact_path.is_dir():
            shutil.rmtree(artifact_path)
            cleaned_artifacts.append(artifact_name)
    if cleaned_artifacts:
        print(f"{Colors.DIM}Cleaned up artifact directories: {', '.join(cleaned_artifacts)}{Colors.RESET}")
    
    print(f"\n{Colors.DIM}{'=' * 60}{Colors.RESET}")
    if not clean and not updated:
        print(f"{Colors.CYAN}Executed notebooks saved to: {OUTPUT_DIR}{Colors.RESET}")
    print(f"{Colors.DIM}Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    
    # Return exit code based on results
    return 1 if results["failed"] else 0


def parse_args():
    """Parse command line arguments."""
    available_folders = ["features", "tutorials", "mini-papers"]
    parser = argparse.ArgumentParser(
        description="Run Jupyter notebooks and report results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available folders: {', '.join(available_folders)}

Examples:
  python run_notebooks.py                    # Run notebooks in {DEFAULT_FOLDERS}
  python run_notebooks.py -f tutorials       # Run only tutorials notebooks
  python run_notebooks.py -f mini-papers     # Run only mini-papers notebooks
  python run_notebooks.py -f features tutorials              # Run features + tutorials
  python run_notebooks.py -f features tutorials mini-papers  # Run all
  python run_notebooks.py --skip vllm_support                # Skip vllm_support notebook
  python run_notebooks.py -s vllm_support remote_execution   # Skip multiple notebooks
  python run_notebooks.py --only cross_prompt early_stopping # Run only specific notebooks
  python run_notebooks.py --clean                            # Don't keep output notebooks
  python run_notebooks.py --update                           # Update source if versions changed
  python run_notebooks.py --force-update                     # Always update source notebooks
        """
    )
    parser.add_argument(
        "-f", "--folders",
        nargs="+",
        default=DEFAULT_FOLDERS,
        metavar="FOLDER",
        help=f"Folders to run notebooks from (default: {' '.join(DEFAULT_FOLDERS)}). "
             f"Available: {', '.join(available_folders)}"
    )
    parser.add_argument(
        "-s", "--skip",
        nargs="+",
        default=[],
        metavar="NOTEBOOK",
        help="Notebook names to skip (without .ipynb extension). "
             "Matches any notebook containing the given name."
    )
    parser.add_argument(
        "-o", "--only",
        nargs="+",
        default=[],
        metavar="NOTEBOOK",
        help="Run only these specific notebooks (without .ipynb extension). "
             "Matches any notebook containing the given name."
    )
    parser.add_argument(
        "-c", "--clean",
        action="store_true",
        help="Delete output notebooks after execution (don't keep them)."
    )
    parser.add_argument(
        "-u", "--update",
        action="store_true",
        help="If all notebooks pass, update source notebooks only when package versions differ."
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="If all notebooks pass, always replace source notebooks with executed outputs."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_notebooks(args.folders, args.skip, args.only, args.clean, args.update, args.force_update))
