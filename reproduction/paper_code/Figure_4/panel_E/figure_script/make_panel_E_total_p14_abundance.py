#!/usr/bin/env python3
"""Render approved frozen whole-section P14 abundance at its upper-right E anchor."""
from paper_paths import Path, lock_record, rendering_script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Figure_4.replicate_revision_20261014.approved_ab_panels import component_main
if __name__ == '__main__':
    component_main('E')
