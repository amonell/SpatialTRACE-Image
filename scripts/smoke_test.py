"""Exercise installed entry points in a new directory; never changes source data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--weights', type=Path, help='Optional verified author release bundle')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    def run(tokens):
        command = [sys.executable, '-m', "crypt_villus_vit.cli", *map(str, tokens)]
        start = time.monotonic()
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        records.append(dict(command=command, seconds=time.monotonic()-start, exit_code=result.returncode))
        (root/f'step_{len(records):02}.log').write_text(result.stdout+result.stderr)
        (root/'acceptance.json').write_text(json.dumps(records, indent=2))
        print(f"step {len(records)}: exit={result.returncode}, {records[-1]['seconds']:.1f}s", flush=True)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
    run(['--help'])
    run(['create-demo', '--output-dir', 'demo'])
    tiny = ['--input-size-px', '64', '--local-crop-px', '64', '--context-crop-px', '128',
            '--fine-crop-px', '32', '--fine-input-size-px', '32', '--embed-dim', '64', '--depth', '1', '--num-heads', '4']
    common = ['--source-manifest', 'demo/sources.csv', '--supervised-manifest', 'demo/labels.csv',
              '--epochs', '2', '--device', args.device]
    run(['train', *common, '--output-dir', 'axis', *tiny])
    run(['predict-cells', '--source-manifest', 'demo/sources.csv', '--cells-csv', 'demo/cells.csv',
         '--checkpoint', 'axis/crypt_villus_vit_model.pt', '--output-dir', 'predictions', '--device', args.device])
    run(['export-qupath', '--predictions', 'predictions/predictions.csv', '--output', 'predictions/cells.geojson'])
    run(['prepare-supervised', '--source-manifest', 'demo/sources.csv', '--cells-csv', 'demo/labels.csv',
         '--output-dir', 'supervised_crops', '--input-size-px', '64', '--local-crop-px', '64',
         '--context-crop-px', '128', '--fine-crop-px', '32', '--fine-input-size-px', '32', '--num-workers', '2'])
    run(['train', '--source-manifest', 'demo/sources.csv',
         '--supervised-manifest', 'supervised_crops/manifest.csv',
         '--prepared-supervised-metadata', 'supervised_crops/metadata.json',
         '--epochs', '1', '--device', args.device, '--output-dir', 'prepared_axis', *tiny])
    run(['predict-cells', '--source-manifest', 'demo/sources.csv', '--cells-csv', 'demo/cells.csv',
         '--checkpoint', 'prepared_axis/crypt_villus_vit_model.pt',
         '--output-dir', 'prepared_predictions', '--device', args.device])
    run(['train', *common, '--output-dir', 'raw_cached_axis', *tiny,
         '--crop-backend', 'cached', '--num-workers', '2', '--ram-crop-cache-mib', '8'])
    run(['predict-cells', '--source-manifest', 'demo/sources.csv', '--cells-csv', 'demo/cells.csv',
         '--checkpoint', 'raw_cached_axis/crypt_villus_vit_model.pt',
         '--output-dir', 'raw_cached_predictions', '--device', args.device])
    run(['prepare-pretraining', '--source-manifest', 'demo/sources.csv', '--cells-csv', 'demo/pretraining_centers.csv',
         '--output-dir', 'paired', '--input-size-px', '64', '--local-crop-px', '64', '--context-crop-px', '128'])
    run(['pretrain-representation', '--prepared-metadata', 'paired/metadata.json', '--output-dir', 'pretrain',
         '--epochs', '1', '--batch-size', '4', '--input-size-px', '64', '--embed-dim', '64',
         '--depth', '1', '--num-heads', '4', '--device', args.device])
    for task, extra in [('fine_tuned_axis', []), ('fine_tuned_peyer', ['--task-type', 'binary_classification', '--target-column', 'peyer_label'])]:
        run(['train', *common, '--pretrained-checkpoint', 'pretrain/dapi_encoder_pretrain.pt', '--output-dir', task, *tiny, *extra])
        run(['predict-cells', '--source-manifest', 'demo/sources.csv', '--cells-csv', 'demo/cells.csv',
             '--checkpoint', f'{task}/crypt_villus_vit_model.pt', '--output-dir', f'{task}_predictions', '--device', args.device])
    if args.weights:
        from crypt_villus_vit.artifacts import registry
        for name, spec in registry().items():
            run(['download', '--model', name, '--from-dir', args.weights.resolve(), '--output-dir', 'weights'])
            if name != 'representation':
                run(['predict-cells', '--source-manifest', 'demo/sources.csv', '--cells-csv', 'demo/cells.csv',
                     '--checkpoint', 'weights/'+spec['filename'], '--output-dir', 'released_'+name, '--device', args.device])
        run(['train', *common, '--pretrained-checkpoint', 'weights/'+registry()['representation']['filename'],
             '--output-dir', 'released_teacher_fine_tuning'])
    import pandas as pd
    rows = pd.read_csv(root/'predictions/predictions.csv')
    assert len(rows) == 8 and rows['source_id'].eq('synthetic_2').all()
    import torch
    best = torch.load(root/'axis/best_crypt_villus_vit_model.pt', weights_only=True)['model_state_dict']
    final = torch.load(root/'axis/crypt_villus_vit_model.pt', weights_only=True)['model_state_dict']
    assert all(torch.equal(best[k], final[k]) for k in best)
    print(root/'acceptance.json', flush=True)


if __name__ == '__main__':
    main()
