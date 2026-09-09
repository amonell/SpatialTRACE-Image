#!/usr/bin/env python3
"""Render the locked annotation-budget comparison as Extended Figure 2F."""
import sys
from paper_paths import Path, lock_record, rendering_script

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from Figure_2_Extended_3.panel_B.figure_script.make_panel_B_matched_annotation_budgets import main

RELEASE_SHA256='ffaa667a34ef1046f64925f03bc629bfd0b12f79b09f50080e904680305206e5'

if __name__=='__main__':
    main(panel='F',release_sha256=RELEASE_SHA256)
