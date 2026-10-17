Panel G reads the fixed 3,373-cell Day 90 replicate-2 window from SATA
`frozen_preprocessing/peyer_representation_v1/image_windows/`.
The active release pointer hash-locks corrected labels, saved probabilities,
unchanged window geometry, and the prespecified full-model/manual-target
classifier. Its backbone and representation-pretrained EMA teacher match the
axis ViT, with one binary output and retained scale embeddings.
Calls use the prespecified 0.5 threshold, not the historical test-optimized 0.56.
Plotting reads the same original microscopy image and performs no inference,
fitting, selection or threshold optimization.
