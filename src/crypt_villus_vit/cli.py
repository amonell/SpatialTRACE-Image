from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from crypt_villus_vit.download import DEFAULT_MODEL_NAME
from crypt_villus_vit.download import download_weights
from crypt_villus_vit.geojson import predictions_to_qupath_geojson
from crypt_villus_vit.model import ModelConfig
from crypt_villus_vit.model import MultitaskDapiVit
from crypt_villus_vit.model import save_model_checkpoint
from crypt_villus_vit.plotting import plot_prediction_scatter
from crypt_villus_vit.pretrain import materialize_pretrained_encoder_checkpoint
from crypt_villus_vit.pretrain import pretrain_image_encoders
from crypt_villus_vit.pretrain import pretrain_prepared_image_encoder
from crypt_villus_vit.prepared import load_prepared_pretraining_arrays
from crypt_villus_vit.predict import predict_cells
from crypt_villus_vit.qupath import load_qupath_centroid_rows
from crypt_villus_vit.sources import load_source_manifest
from crypt_villus_vit.train import train_model
from crypt_villus_vit.xenium_manifest import build_xenium_manifests


def _add_common_prediction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-type", choices=("axis_regression", "binary_classification"), default=None)
    parser.add_argument("--prediction-column", type=str, default=None)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--local-crop-px", type=int, default=None)
    parser.add_argument("--context-crop-px", type=int, default=None)
    parser.add_argument("--fine-crop-px", type=int, default=None)
    parser.add_argument("--reference-pixel-size-um", type=float, default=None)
    parser.add_argument("--crop-backend", choices=("reference", "cached", "rust"), default="reference",
                        help="cached reuses decoded TIFF tiles; rust also uses native normalization.")
    parser.add_argument("--tile-cache-mib", type=int, default=256,
                        help="Decoded tile cache limit per worker for cached/rust backends.")
    parser.add_argument("--spatial-order", action=argparse.BooleanOptionalAction, default=None,
                        help="Group nearby cells for I/O; defaults on for cached/rust. Output order is unchanged.")
    parser.add_argument("--no-scatter", action="store_true", help="Skip the optional output scatter image.")


def command_predict_qupath(args: argparse.Namespace) -> None:
    sources = load_source_manifest(args.source_manifest)
    source_id = args.source_id
    if source_id is None:
        if len(sources) != 1:
            raise ValueError("--source-id is required when the source manifest contains multiple sources.")
        source_id = next(iter(sources))
    pixel_size = args.pixel_size_um if args.pixel_size_um is not None else sources[source_id].pixel_size_um
    rows, import_summary = load_qupath_centroid_rows(
        qupath_input_path=args.qupath_csv,
        source_id=source_id,
        class_filter=args.class_filter,
        class_match_mode=args.class_match_mode,
        row_id_prefix=args.row_id_prefix,
        pixel_size_um=pixel_size,
    )
    summary = predict_cells(
        rows,
        sources=sources,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        task_type=args.task_type,
        prediction_column=args.prediction_column,
        classification_threshold=args.classification_threshold,
        local_crop_px=args.local_crop_px,
        context_crop_px=args.context_crop_px,
        fine_crop_px=args.fine_crop_px,
        reference_pixel_size_um=args.reference_pixel_size_um,
        crop_backend=args.crop_backend,
        tile_cache_mib=args.tile_cache_mib,
        spatial_order=args.spatial_order,
    )
    if summary["task_type"] == "axis_regression" and not args.no_scatter:
        plot_prediction_scatter(Path(summary["prediction_csv"]), args.output_dir / "prediction_scatter.png")
    (args.output_dir / "qupath_import_summary.json").write_text(
        __import__("json").dumps(import_summary.to_dict(), indent=2)
    )
    print(summary["prediction_csv"])


def command_predict_cells(args: argparse.Namespace) -> None:
    sources = load_source_manifest(args.source_manifest)
    rows = pd.read_csv(args.cells_csv)
    summary = predict_cells(
        rows,
        sources=sources,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        task_type=args.task_type,
        prediction_column=args.prediction_column,
        classification_threshold=args.classification_threshold,
        local_crop_px=args.local_crop_px,
        context_crop_px=args.context_crop_px,
        fine_crop_px=args.fine_crop_px,
        reference_pixel_size_um=args.reference_pixel_size_um,
        crop_backend=args.crop_backend,
        tile_cache_mib=args.tile_cache_mib,
        spatial_order=args.spatial_order,
    )
    if summary["task_type"] == "axis_regression" and not args.no_scatter:
        plot_prediction_scatter(Path(summary["prediction_csv"]), args.output_dir / "prediction_scatter.png")
    print(summary["prediction_csv"])


def command_train(args: argparse.Namespace) -> None:
    sources = load_source_manifest(args.source_manifest)
    rows = pd.read_csv(args.supervised_manifest)
    summary = train_model(
        rows,
        sources=sources,
        output_dir=args.output_dir,
        config=ModelConfig(
            local_crop_px=args.local_crop_px,
            context_crop_px=args.context_crop_px,
            fine_crop_px=args.fine_crop_px,
            input_size_px=args.input_size_px,
            fine_input_size_px=args.fine_input_size_px,
            patch_size_px=args.patch_size_px,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            use_local_branch=not args.no_local_branch,
            use_context_branch=not args.no_context_branch,
            use_fine_branch=not args.no_fine_branch,
            local_readout=args.local_readout,
            context_readout=args.context_readout,
            encoder_architecture=args.encoder_architecture,
            retain_scale_embeddings=args.encoder_architecture == 'shared_scale_aware',
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        task_type=args.task_type,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
        positive_class_weight=args.positive_class_weight,
        classification_threshold=args.classification_threshold,
        target_axis_column=args.target_axis_column,
        target_epithelial_column=args.target_epithelial_column,
        initial_checkpoint=args.initial_checkpoint,
        pretrained_checkpoint=args.pretrained_checkpoint,
        prepared_supervised_metadata=args.prepared_supervised_metadata,
        augment_intensity_probability=args.augment_intensity_probability,
        augment_brightness_delta=args.augment_brightness_delta,
        augment_contrast_range=tuple(args.augment_contrast_range),
        augment_gaussian_blur_sigma_range=tuple(args.augment_gaussian_blur_sigma_range),
        gradient_clip_norm=args.gradient_clip_norm,
        reference_pixel_size_um=args.reference_pixel_size_um,
        freeze_mode=args.freeze_mode,
    )
    print(summary["checkpoint_path"])


def command_pretrain(args: argparse.Namespace) -> None:
    config = ModelConfig(
        local_crop_px=args.local_crop_px,
        context_crop_px=args.context_crop_px,
        fine_crop_px=args.fine_crop_px,
        input_size_px=args.input_size_px,
        fine_input_size_px=args.fine_input_size_px,
        patch_size_px=args.patch_size_px,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        use_local_branch=not args.no_local_branch,
        use_context_branch=not args.no_context_branch,
        use_fine_branch=False,
        local_readout=args.local_readout,
        context_readout=args.context_readout,
    )
    if args.materialize_only:
        if args.initial_checkpoint is None:
            raise ValueError("--materialize-only requires --initial-checkpoint.")
        summary = materialize_pretrained_encoder_checkpoint(
            source_checkpoint=args.initial_checkpoint,
            output_dir=args.output_dir,
            config=config,
            source_history_csv=args.initial_history_csv,
            metadata={
                "source_manifest": str(args.source_manifest),
                "supervised_manifest": str(args.supervised_manifest),
            },
        )
        print(summary["checkpoint_path"])
        return
    if args.prepared_pretraining_metadata is not None:
        arrays = load_prepared_pretraining_arrays(args.prepared_pretraining_metadata)
        manifest_path = args.supervised_manifest or arrays.manifest_path
        rows = pd.read_csv(manifest_path)
        summary = pretrain_prepared_image_encoder(
            rows,
            prepared_pretraining_metadata=args.prepared_pretraining_metadata,
            output_dir=args.output_dir,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            mask_fraction=args.mask_fraction,
            validation_fraction=args.validation_fraction,
            device=args.device,
            seed=args.seed,
            num_workers=args.num_workers,
            augment_intensity_probability=args.augment_intensity_probability,
            augment_brightness_delta=args.augment_brightness_delta,
            augment_contrast_range=tuple(args.augment_contrast_range),
            augment_gaussian_blur_sigma_range=tuple(args.augment_gaussian_blur_sigma_range),
            gradient_clip_norm=args.gradient_clip_norm,
        )
        print(summary["checkpoint_path"])
        return
    if args.source_manifest is None or args.supervised_manifest is None:
        raise ValueError(
            "Raw-image pretraining requires --source-manifest and --supervised-manifest."
        )
    sources = load_source_manifest(args.source_manifest)
    rows = pd.read_csv(args.supervised_manifest)
    summary = pretrain_image_encoders(
        rows,
        sources=sources,
        output_dir=args.output_dir,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mask_fraction=args.mask_fraction,
        validation_fraction=args.validation_fraction,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        augment_intensity_probability=args.augment_intensity_probability,
        augment_brightness_delta=args.augment_brightness_delta,
        augment_contrast_range=tuple(args.augment_contrast_range),
        augment_gaussian_blur_sigma_range=tuple(args.augment_gaussian_blur_sigma_range),
        gradient_clip_norm=args.gradient_clip_norm,
        reference_pixel_size_um=args.reference_pixel_size_um,
    )
    print(summary["checkpoint_path"])


def command_build_xenium_manifests(args: argparse.Namespace) -> None:
    summary = build_xenium_manifests(
        h5ad_path=args.h5ad,
        output_dir=args.output_dir,
        image_root=args.image_root,
        image_glob=args.image_glob,
        batch_column=args.batch_column,
        spatial_key=args.spatial_key,
        spatial_units=args.spatial_units,
        axis_column=args.axis_column,
        epithelial_column=args.epithelial_column,
        epithelial_clip_max=args.epithelial_clip_max,
        pixel_size_um=args.pixel_size_um,
        channel_index=args.channel_index,
        max_cells_per_source=args.max_cells_per_source,
        max_train_cells_per_source=args.max_train_cells_per_source,
        max_validation_cells_per_source=args.max_validation_cells_per_source,
        validation_fraction=args.validation_fraction,
        validation_mode=args.validation_mode,
        seed=args.seed,
    )
    print(summary["summary_json"])


def command_export_qupath(args: argparse.Namespace) -> None:
    print(predictions_to_qupath_geojson(args.predictions, args.output))


def command_download_weights(args: argparse.Namespace) -> None:
    from .artifacts import download
    print(download(args.model, args.output_dir, source_dir=args.from_dir, base_url=args.base_url))


def command_create_demo_checkpoint(args: argparse.Namespace) -> None:
    config = ModelConfig(
        local_crop_px=64,
        context_crop_px=128,
        fine_crop_px=32,
        input_size_px=64,
        fine_input_size_px=32,
        patch_size_px=16,
        embed_dim=64,
        depth=1,
        num_heads=4,
    )
    model = MultitaskDapiVit(config)
    save_model_checkpoint(model, args.output, metadata={"purpose": "demo smoke-test checkpoint"})
    print(args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spatialtrace-image")
    parser.add_argument('--threads', type=int, default=4, help='CPU threads per process.')
    sub = parser.add_subparsers(dest="command", required=True)

    predict_qupath = sub.add_parser("predict-qupath", help="Predict on a QuPath detection/measurement CSV.")
    _add_common_prediction_args(predict_qupath)
    predict_qupath.add_argument("--qupath-csv", type=Path, required=True)
    predict_qupath.add_argument("--source-id", type=str, default=None)
    predict_qupath.add_argument("--class-filter", type=str, default="")
    predict_qupath.add_argument("--class-match-mode", type=str, default="exact")
    predict_qupath.add_argument("--row-id-prefix", type=str, default="cell")
    predict_qupath.add_argument("--pixel-size-um", type=float, default=None)
    predict_qupath.set_defaults(func=command_predict_qupath)

    predict_cells = sub.add_parser("predict-cells", help="Predict on a normalized cell manifest CSV.")
    _add_common_prediction_args(predict_cells)
    predict_cells.add_argument("--cells-csv", type=Path, required=True)
    predict_cells.set_defaults(func=command_predict_cells)

    train = sub.add_parser("train", help="Train a multitask model from a supervised cell manifest.")
    train.add_argument("--source-manifest", type=Path, required=True)
    train.add_argument("--supervised-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--device", type=str, default="cuda")
    train.add_argument("--validation-fraction", type=float, default=0.0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--task-type", choices=("axis_regression", "binary_classification"), default="axis_regression")
    train.add_argument("--target-column", type=str, default=None)
    train.add_argument("--prediction-column", type=str, default=None)
    train.add_argument("--positive-class-weight", type=str, default="auto")
    train.add_argument("--classification-threshold", type=float, default=0.5)
    train.add_argument("--target-axis-column", type=str, default="target_axis")
    train.add_argument("--target-epithelial-column", type=str, default="epithelial_distance_clipped_1p0")
    train.add_argument("--initial-checkpoint", type=Path, default=None)
    train.add_argument("--pretrained-checkpoint", type=Path, default=None)
    train.add_argument("--prepared-supervised-metadata", type=Path, default=None)
    train.add_argument("--augment-intensity-probability", type=float, default=0.0)
    train.add_argument("--augment-brightness-delta", type=float, default=0.0)
    train.add_argument("--augment-contrast-range", type=float, nargs=2, default=(1.0, 1.0), metavar=("MIN", "MAX"))
    train.add_argument(
        "--augment-gaussian-blur-sigma-range",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("MIN", "MAX"),
    )
    train.add_argument("--gradient-clip-norm", type=float, default=None)
    train.add_argument("--reference-pixel-size-um", type=float, default=0.325)
    train.add_argument("--freeze-mode", choices=("none", "heads", "last1"), default="none")
    train.add_argument("--local-crop-px", type=int, default=512)
    train.add_argument("--context-crop-px", type=int, default=2048)
    train.add_argument("--fine-crop-px", type=int, default=128)
    train.add_argument("--input-size-px", type=int, default=256)
    train.add_argument("--fine-input-size-px", type=int, default=128)
    train.add_argument("--patch-size-px", type=int, default=16)
    train.add_argument("--embed-dim", type=int, default=256)
    train.add_argument("--depth", type=int, default=6)
    train.add_argument("--num-heads", type=int, default=8)
    train.add_argument("--local-readout", type=str, default="center_4x")
    train.add_argument("--context-readout", type=str, default="center_4x")
    train.add_argument('--encoder-architecture', choices=('shared_scale_aware', 'separate_patch'), default='shared_scale_aware')
    train.add_argument("--no-local-branch", action="store_true")
    train.add_argument("--no-context-branch", action="store_true")
    train.add_argument("--no-fine-branch", action="store_true")
    train.set_defaults(func=command_train)

    pretrain = sub.add_parser("pretrain", help="Masked-image pretraining for DAPI local/context encoders.")
    pretrain.add_argument("--source-manifest", type=Path, default=None)
    pretrain.add_argument("--supervised-manifest", type=Path, default=None)
    pretrain.add_argument("--prepared-pretraining-metadata", type=Path, default=None)
    pretrain.add_argument("--output-dir", type=Path, required=True)
    pretrain.add_argument("--epochs", type=int, default=2)
    pretrain.add_argument("--batch-size", type=int, default=8)
    pretrain.add_argument("--learning-rate", type=float, default=1e-4)
    pretrain.add_argument("--weight-decay", type=float, default=1e-4)
    pretrain.add_argument("--mask-fraction", type=float, default=0.30)
    pretrain.add_argument("--validation-fraction", type=float, default=0.0)
    pretrain.add_argument("--device", type=str, default="cuda")
    pretrain.add_argument("--seed", type=int, default=0)
    pretrain.add_argument("--num-workers", type=int, default=0)
    pretrain.add_argument("--augment-intensity-probability", type=float, default=0.0)
    pretrain.add_argument("--augment-brightness-delta", type=float, default=0.0)
    pretrain.add_argument(
        "--augment-contrast-range",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
        metavar=("MIN", "MAX"),
    )
    pretrain.add_argument(
        "--augment-gaussian-blur-sigma-range",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("MIN", "MAX"),
    )
    pretrain.add_argument("--gradient-clip-norm", type=float, default=None)
    pretrain.add_argument("--reference-pixel-size-um", type=float, default=None)
    pretrain.add_argument("--initial-checkpoint", type=Path, default=None)
    pretrain.add_argument("--initial-history-csv", type=Path, default=None)
    pretrain.add_argument("--materialize-only", action="store_true")
    pretrain.add_argument("--local-crop-px", type=int, default=512)
    pretrain.add_argument("--context-crop-px", type=int, default=2048)
    pretrain.add_argument("--fine-crop-px", type=int, default=128)
    pretrain.add_argument("--input-size-px", type=int, default=256)
    pretrain.add_argument("--fine-input-size-px", type=int, default=128)
    pretrain.add_argument("--patch-size-px", type=int, default=16)
    pretrain.add_argument("--embed-dim", type=int, default=256)
    pretrain.add_argument("--depth", type=int, default=6)
    pretrain.add_argument("--num-heads", type=int, default=8)
    pretrain.add_argument("--local-readout", type=str, default="mean")
    pretrain.add_argument("--context-readout", type=str, default="mean")
    pretrain.add_argument("--no-local-branch", action="store_true")
    pretrain.add_argument("--no-context-branch", action="store_true")
    pretrain.set_defaults(func=command_pretrain)

    manifests = sub.add_parser("build-xenium-manifests", help="Export source/cell manifests from a Xenium h5ad.")
    manifests.add_argument("--h5ad", type=Path, required=True)
    manifests.add_argument("--output-dir", type=Path, required=True)
    manifests.add_argument("--image-root", type=Path, required=True)
    manifests.add_argument("--image-glob", type=str, default="**/xenium_output/morphology_mip.ome.tif")
    manifests.add_argument("--batch-column", type=str, default="batch")
    manifests.add_argument("--spatial-key", type=str, default="X_spatial")
    manifests.add_argument("--spatial-units", type=str, default="pixels", choices=("pixels", "microns"))
    manifests.add_argument("--axis-column", type=str, default="crypt_villi_axis")
    manifests.add_argument("--epithelial-column", type=str, default="epithelial_distance_clipped")
    manifests.add_argument("--epithelial-clip-max", type=float, default=None)
    manifests.add_argument("--pixel-size-um", type=float, default=None)
    manifests.add_argument("--channel-index", type=int, default=0)
    manifests.add_argument("--max-cells-per-source", type=int, default=None)
    manifests.add_argument("--max-train-cells-per-source", type=int, default=None)
    manifests.add_argument("--max-validation-cells-per-source", type=int, default=None)
    manifests.add_argument("--validation-fraction", type=float, default=0.20)
    manifests.add_argument("--validation-mode", type=str, default="section", choices=("section", "cell"))
    manifests.add_argument("--seed", type=int, default=0)
    manifests.set_defaults(func=command_build_xenium_manifests)

    export = sub.add_parser("export-qupath", help="Export prediction points as QuPath-readable GeoJSON.")
    export.add_argument("--predictions", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=command_export_qupath)

    download = sub.add_parser("download", aliases=['download-weights'], help="Download a checksum-verified release checkpoint.")
    download.add_argument("--model", choices=('xenium', 'if', 'peyer', 'representation'), default='xenium')
    download.add_argument("--output-dir", type=Path, default=Path("weights"))
    download.add_argument('--from-dir', type=Path)
    download.add_argument('--base-url')
    download.set_defaults(func=command_download_weights)

    demo = sub.add_parser("create-demo-checkpoint", help="Create a tiny random checkpoint for demos/tests.")
    demo.add_argument("--output", type=Path, required=True)
    demo.set_defaults(func=command_create_demo_checkpoint)
    from .release_workflows import add_commands
    add_commands(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    import torch
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads)
    args.func(args)
    from .provenance import record_cli
    record_cli(args, argv)


if __name__ == "__main__":
    main()
