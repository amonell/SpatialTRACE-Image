"""Fail before archiving active components when a scientific repair is unfinished."""
import json
from paper_paths import Path, lock_record, rendering_script


def unresolved_for(packages, output_root):
    packages = set(packages)
    output_root = Path(output_root)
    blockers = []
    if packages & {'Figure_1', 'Figure_4', 'Figure_4_Extended'}:
        release = output_root / 'Figure_4/stored_outputs/scientific_repair_v1/evaluation/release_manifest.json'
        if not release.is_file():
            blockers.append(f'Repaired IF release is not complete: {release}. The previous IF outputs must not be silently reused.')
    if 'Figure_2_Extended' in packages:
        benchmark = output_root / 'Figure_2_Extended/stored_outputs/runtime_benchmark_v1/completed.json'
        if not benchmark.is_file():
            blockers.append(f'Matched runtime benchmark has not completed: {benchmark}. C/D can be rendered explicitly using their package runner.')
    repaired = output_root / 'frozen_preprocessing/peyer_label_repair_v1'
    if packages & {'Figure_1', 'Figure_2_Extended_2'}:
        graph = repaired / 'gat/gat_summary.json'
        if not graph.is_file() or json.loads(graph.read_text()).get('status') != 'FROZEN_CORRECTED_LABEL_GRAPH':
            blockers.append(f'Corrected-label Peyer graph classifier is not frozen: {graph}')
    if packages & {'Figure_3_Extended', 'Figure_4_Extended'}:
        from frozen_preprocessing.peyer_representation_v1.release import load_release, require_record
        try:
            _, classifier = load_release()
            require_record(classifier['checkpoint'])
            require_record(classifier['window_release'])
            require_record(classifier['comparison'])
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            blockers.append(f'Representation-pretrained Peyer classifier release is incomplete or changed: {error}')
    return blockers
