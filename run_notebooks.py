#!/usr/bin/env python3
"""
Script to run all Jupyter notebooks using papermill.
Catches and reports any errors that occur during execution.

By default, runs notebooks in both 'features' and 'tutorials' folders.
Use --folders to specify individual folders to run.
"""

import argparse
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


def get_version_info() -> str:
    """Generate markdown content with package versions."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    return f"""
**Execution Timestamp:** {timestamp}

**System Info:**

| Package | Version |
|---------|---------|
| **nnsight** | **{nnsight.__version__}** |
| torch | {torch.__version__} |
| transformers | {transformers.__version__} |
"""


def add_version_cell(notebook_path: Path) -> None:
    """Add a markdown cell with version info to the beginning of a notebook."""
    nb = nbformat.read(notebook_path, as_version=4)
    
    version_cell = nbformat.v4.new_markdown_cell(get_version_info())
    nb.cells.insert(0, version_cell)
    
    nbformat.write(nb, notebook_path)


def run_notebooks(folders: list[str], skip: list[str] = None, clean: bool = False):
    """Run all notebooks in the specified folders and collect results."""
    skip = skip or []
    output_files = []  # Track output files for cleanup
    
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
            
            if not clean:
                add_version_cell(output_path)
            output_files.append(output_path)
            print(f"{Colors.GREEN}✓ PASSED: {display_name}{Colors.RESET}")
            results["passed"].append(display_name)
            
        except PapermillExecutionError as e:
            # Still add version info to failed notebooks (unless clean mode)
            if output_path.exists():
                if not clean:
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
    
    # Clean up output files if requested
    if clean and output_files:
        for output_file in output_files:
            if output_file.exists():
                output_file.unlink()
        print(f"\n{Colors.DIM}Cleaned up {len(output_files)} output notebook(s){Colors.RESET}")
    
    print(f"\n{Colors.DIM}{'=' * 60}{Colors.RESET}")
    if not clean:
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
  python run_notebooks.py --clean                            # Don't keep output notebooks
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
        "-c", "--clean",
        action="store_true",
        help="Delete output notebooks after execution (don't keep them)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_notebooks(args.folders, args.skip, args.clean))
