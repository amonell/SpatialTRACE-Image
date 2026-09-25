"""Portable tutorials and representation-pretraining entry points."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .artifacts import registry
from .model import ModelConfig
from .prepared import load_prepared_pretraining_arrays


def empty_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_demo(args):
    root = empty_directory(args.output_dir)
    rng = np.random.default_rng(17)
    y, x = np.mgrid[:256, :256]
    sources, rows = [], []
    for section, split in enumerate(('train', 'validation', 'test')):
        source_id = f'synthetic_{section}'
        image = rng.normal(100, 5, size=(256, 256))
        for cx, cy in rng.uniform(16, 240, size=(80, 2)):
            image += 250 * np.exp(-((x-cx)**2 + (y-cy)**2)/20)
        np.save(root / f'{source_id}.npy', image.astype(np.float32))
        sources.append(dict(source_id=source_id, section_id=source_id,
                            image_path=f'{source_id}.npy', pixel_size_um=.325, channel_index=0))
        for index, (cx, cy) in enumerate(rng.uniform(40, 216, size=(8, 2))):
            rows.append(dict(cell_id=f'{source_id}_{index}', source_id=source_id,
                             centroid_x_fullres_px=cx, centroid_y_fullres_px=cy,
                             target_axis=cy/256, epithelial_distance_clipped_1p0=cx/256,
                             peyer_label=index % 2, split=split))
    pd.DataFrame(sources).to_csv(root/'sources.csv', index=False)
    table = pd.DataFrame(rows)
    table.to_csv(root/'labels.csv', index=False)
    table.loc[table.split == 'test', ['cell_id', 'source_id', 'centroid_x_fullres_px', 'centroid_y_fullres_px']].to_csv(root/'cells.csv', index=False)
    table.loc[table.split != 'test'].to_csv(root/'pretraining_centers.csv', index=False)
    (root/'README.txt').write_text('Entirely synthetic microscopy and labels. Tests software execution, not anatomical accuracy. Test section is excluded from pretraining_centers.csv.\n')
    print(root)


def prepare_pairs(args):
    from .prepare_crops import prepare_crops
    return prepare_crops(args)


def prepare_supervised(args):
    from .prepare_crops import prepare_crops
    return prepare_crops(args, supervised=True)


def add_preparation_arguments(parser):
    from argparse import BooleanOptionalAction
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--cells-csv', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--local-crop-px', default=512, type=int)
    parser.add_argument('--context-crop-px', default=2048, type=int)
    parser.add_argument('--input-size-px', default=256, type=int)
    parser.add_argument('--reference-pixel-size-um', default=.325, type=float)
    parser.add_argument('--rows-per-shard', default=1024, type=int)
    parser.add_argument('--crop-backend', choices=('reference', 'cached', 'rust'), default='cached')
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--batch-size', default=128, type=int, help='Cells per preparation batch, not training batch.')
    parser.add_argument('--tile-cache-mib', default=256, type=int, help='Decoded tile cache per worker.')
    parser.add_argument('--spatial-order', action=BooleanOptionalAction, default=None,
                        help='Read nearby cells together; saved row order remains unchanged.')


def representation_pretrain(args):
    from .representation_pretrain import pretrain_paired_scale_teacher_student
    empty_directory(args.output_dir)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install cu126 or use --device cpu')
    metadata = load_prepared_pretraining_arrays(args.prepared_metadata)
    crop_settings = json.loads(args.prepared_metadata.read_text()).get('crop_configuration', {})
    if crop_settings and crop_settings['input_size_px'] != args.input_size_px:
        raise ValueError('Prepared input_size_px differs from pretraining input_size_px')
    config = ModelConfig(local_crop_px=crop_settings.get('local_crop_px', 512),
                         context_crop_px=crop_settings.get('context_crop_px', 2048),
                         input_size_px=args.input_size_px, patch_size_px=args.patch_size_px,
                         embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                         use_fine_branch=False, encoder_architecture='shared_scale_aware',
                         retain_scale_embeddings=True, local_readout='center_4x', context_readout='center_4x')
    rows = pd.read_csv(metadata.manifest_path)
    if 'split' in rows and rows.split.astype(str).str.lower().eq('test').any():
        raise ValueError('Test rows are not valid pretraining inputs')
    summary = pretrain_paired_scale_teacher_student(rows, prepared_pretraining_metadata=args.prepared_metadata,
        output_dir=args.output_dir, config=config, epochs=args.epochs, batch_size=args.batch_size,
        validation_fraction=args.validation_fraction, device=args.device, seed=args.seed,
        readout='center_4x', resume=False)
    print(summary['checkpoint_path'])


def add_commands(sub):
    models = sub.add_parser('models', help='List verified release artifact identities.')
    models.set_defaults(func=lambda args: print(json.dumps(registry(), indent=2)))
    demo = sub.add_parser('create-demo', help='Generate a self-contained synthetic tutorial.')
    demo.add_argument('--output-dir', required=True, type=Path)
    demo.set_defaults(func=create_demo)
    prepare = sub.add_parser('prepare-pretraining', help='Prepare matched local/context crops from own images.')
    add_preparation_arguments(prepare)
    prepare.set_defaults(func=prepare_pairs)
    supervised = sub.add_parser('prepare-supervised', help='Prepare reusable local/context/fine training crops.')
    add_preparation_arguments(supervised)
    supervised.add_argument('--fine-crop-px', default=128, type=int)
    supervised.add_argument('--fine-input-size-px', default=128, type=int)
    from argparse import BooleanOptionalAction
    supervised.add_argument('--fine-branch', action=BooleanOptionalAction, default=True)
    supervised.set_defaults(func=prepare_supervised)
    pretrain = sub.add_parser('pretrain-representation', help='Train the paired-scale masked EMA representation model.')
    pretrain.add_argument('--prepared-metadata', required=True, type=Path)
    pretrain.add_argument('--output-dir', required=True, type=Path)
    pretrain.add_argument('--epochs', default=35, type=int)
    pretrain.add_argument('--batch-size', default=64, type=int)
    pretrain.add_argument('--input-size-px', default=256, type=int)
    pretrain.add_argument('--patch-size-px', default=16, type=int)
    pretrain.add_argument('--embed-dim', default=256, type=int)
    pretrain.add_argument('--depth', default=6, type=int)
    pretrain.add_argument('--num-heads', default=8, type=int)
    pretrain.add_argument('--validation-fraction', default=.1, type=float)
    pretrain.add_argument('--seed', default=7, type=int)
    pretrain.add_argument('--device', default='cpu')
    pretrain.set_defaults(func=representation_pretrain)
