#!/usr/bin/env python3
"""Refresh six candidate pages from the hash-locked post-removal cohort."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

from render_early_preview import Snapshot, regional_comparison, imaps, compositions
from render_replicate_extras import new_maps, fields_by_gate, background_imaps
from render_review_panels import config, sha


class FilteredSnapshot(Snapshot):
    def footer(self):
        return 'After tissue removals · 4 DMSO and 5 RARi mice · Original pair retained · Tissue 08 excluded'

    def save(self, figure, stem):
        for text in figure.texts:
            if text.get_text() == 'Each bar represents one mouse; all cells are included.':
                text.set_text('Each bar represents one mouse; all retained cells are included.')
        super().save(figure, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--cohort-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.cohort) == args.cohort_sha256
    config()
    data = FilteredSnapshot(args.cohort.parent, args.output)
    assert data.manifest['n_cells'] == 1568177 and data.manifest['n_p14'] == 43424
    assert data.manifest['mice_per_condition'] == {'DMSO': 4, 'RARi': 5}
    for render in [regional_comparison, imaps, compositions, new_maps, fields_by_gate, background_imaps]:
        render(data)
    paths = [Path(__file__), *[Path(__file__).with_name(n) for n in
             ['render_early_preview.py', 'render_replicate_extras.py', 'render_review_panels.py']]]
    record = lambda p: {'path': str(Path(p).resolve()), 'sha256': sha(p)}
    manifest = {'status': 'rendered_pending_visual_QA', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'command': [sys.executable, *sys.argv], 'python': platform.python_version(),
                'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'matplotlib', 'pyarrow']},
                'random_seed': None, 'git_state': 'Not a Git worktree',
                'inputs': [record(args.cohort), *data.inputs], 'code': [record(p) for p in paths],
                'outputs': data.outputs, 'warnings': data.manifest['warnings'],
                'analysis_performed_during_render': False, 'paper_figures_modified': False}
    (args.output/'rendering_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
