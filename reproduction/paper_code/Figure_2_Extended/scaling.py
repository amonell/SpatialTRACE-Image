"""Read the same frozen section-disjoint experiment as main Figure 2e."""
import json
import pandas as pd
from common import require_file

METRICS_SHA = '462e31be3c8c6cfe0ec7aff09f282bfe5bb685f16f151bd3484f3478e75dcf94'
SPLIT_SHA = 'e1779c3caab90164eea05ba38d2dc93c3432ad845711fd7428f281bd52eaa0f8'
COUNTS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]


def load_scaling(metrics, split_manifest):
    source = require_file(metrics, METRICS_SHA)
    split_path = require_file(split_manifest, SPLIT_SHA)
    split = json.loads(split_path.read_text())
    frame = pd.read_csv(source, sep='\t')
    if len(frame) != 160 or sorted(frame.n_train_villi.unique()) != COUNTS:
        raise RuntimeError('Expected the complete 160-run section-disjoint experiment')
    if frame.duplicated(['axis', 'n_train_villi', 'fold_id']).any():
        raise RuntimeError('Duplicate scaling run')
    if not frame.groupby(['axis', 'n_train_villi']).test_section.nunique().eq(8).all():
        raise RuntimeError('Each point requires eight held-out sections')
    if split['aggregation_unit'] != 'held-out tissue section':
        raise RuntimeError('Incorrect evaluation unit')
    for fold in split['folds']:
        if fold['test_section'] in fold['training_sections'] or fold['validation_section'] in fold['training_sections'] or fold['test_section'] == fold['validation_section']:
            raise RuntimeError('Section overlap')
    return source, split_path, frame
