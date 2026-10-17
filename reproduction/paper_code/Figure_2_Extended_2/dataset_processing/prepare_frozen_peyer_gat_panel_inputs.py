#!/usr/bin/env python3
"""Forward explicit input preparation to the frozen corrected-label pipeline.

The previous 0.88-threshold preparer is preserved in the scientific-repair
original_sources snapshot. This entry point never fits or selects a model.
"""
import runpy
import sys
from paper_paths import Path, lock_record, rendering_script

if __name__ == "__main__":
    pipeline = Path(__file__).resolve().parents[2] / "frozen_preprocessing/peyer_label_repair_v1"
    sys.path.insert(0, str(pipeline))
    runpy.run_path(str(pipeline / "prepare_graph_panels.py"), run_name="__main__")
