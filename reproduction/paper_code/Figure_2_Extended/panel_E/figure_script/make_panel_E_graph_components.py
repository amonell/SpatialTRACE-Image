#!/usr/bin/env python3
"""Render the frozen graph-component comparison as Extended Figure 2E."""
import sys
from paper_paths import Path, lock_record, rendering_script

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Figure_2_Extended_3.panel_A.figure_script.make_panel_A_graph_components import main

if __name__ == '__main__':
    main(panel='E')
