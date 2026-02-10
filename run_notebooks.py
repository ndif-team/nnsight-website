#!/usr/bin/env python3
"""
Script to run all Jupyter notebooks in the features folder using papermill.
Catches and reports any errors that occur during execution.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import papermill as pm
from papermill.exceptions import PapermillExecutionError


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
NOTEBOOKS_DIR = Path(__file__).parent / "source" / "notebooks" / "features"
OUTPUT_DIR = Path(__file__).parent / "notebook_outputs"


def run_notebooks():
    """Run all notebooks in the features folder and collect results."""
    
    # Create output directory for executed notebooks
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find all notebook files
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    
    if not notebooks:
        print(f"{Colors.YELLOW}No notebooks found in {NOTEBOOKS_DIR}{Colors.RESET}")
        return 1
    
    print(f"{Colors.BOLD}Found {len(notebooks)} notebooks to run{Colors.RESET}")
    print(f"{Colors.DIM}{'=' * 60}{Colors.RESET}")
    
    results = {
        "passed": [],
        "failed": [],
        "errors": {}
    }
    
    for notebook_path in notebooks:
        notebook_name = notebook_path.name
        output_path = OUTPUT_DIR / f"executed_{notebook_name}"
        
        print(f"\n{Colors.CYAN}▶ Running: {notebook_name}{Colors.RESET}")
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
            
            print(f"{Colors.GREEN}✓ PASSED: {notebook_name}{Colors.RESET}")
            results["passed"].append(notebook_name)
            
        except PapermillExecutionError as e:
            print(f"{Colors.RED}✗ FAILED: {notebook_name}{Colors.RESET}")
            print(f"{Colors.RED}  Cell {e.cell_index}: {Colors.YELLOW}{e.ename}: {e.evalue}{Colors.RESET}")
            results["failed"].append(notebook_name)
            results["errors"][notebook_name] = {
                "cell_index": e.cell_index,
                "error_name": e.ename,
                "error_value": e.evalue,
                "traceback": e.traceback,
            }
            
        except Exception as e:
            print(f"{Colors.RED}✗ FAILED: {notebook_name}{Colors.RESET}")
            print(f"{Colors.RED}  Unexpected error: {Colors.YELLOW}{type(e).__name__}: {e}{Colors.RESET}")
            results["failed"].append(notebook_name)
            results["errors"][notebook_name] = {
                "cell_index": None,
                "error_name": type(e).__name__,
                "error_value": str(e),
                "traceback": None,
            }
    
    # Print summary
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}{Colors.RESET}")
    print(f"Total notebooks: {len(notebooks)}")
    print(f"{Colors.GREEN}Passed: {len(results['passed'])}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {len(results['failed'])}{Colors.RESET}")
    
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
    
    print(f"\n{Colors.DIM}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.CYAN}Executed notebooks saved to: {OUTPUT_DIR}{Colors.RESET}")
    print(f"{Colors.DIM}Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    
    # Return exit code based on results
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(run_notebooks())
