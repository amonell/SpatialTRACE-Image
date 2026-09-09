import numpy as np
import pandas as pd
import pytest
import tifffile
from crypt_villus_vit.production_crops import ImagePyramid, normalize_resize_quantize
from crypt_villus_vit.sources import SourceSpec
from crypt_villus_vit.train import split_supervised_manifest, prepare_supervised_manifest, _binary_classification_metrics


def test_pyramid_base_and_overview_are_arrays(tmp_path):
    image = np.arange(128*128, dtype=np.uint16).reshape(128, 128)
    path = tmp_path/'pyramid.ome.tif'
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(image, subifds=1, metadata={'axes': 'YX'}, photometric='minisblack')
        writer.write(image[::2, ::2], subfiletype=1, photometric='minisblack')
    backend = ImagePyramid(SourceSpec('a', path))
    assert np.array_equal(backend.read(0, 1, 3, 2, 5), image[1:3, 2:5])
    assert np.array_equal(backend.read(1, 1, 3, 2, 5), image[::2, ::2][1:3, 2:5])
    assert backend.choose_level(64, 32) == 1
    assert backend.choose_level(32, 32) == 0
    value = backend.crop(64, 64, 32, 32)
    assert np.array_equal(value, normalize_resize_quantize(image[48:80, 48:80], 32))
    assert np.isfinite(backend.crop(1, 1, 32, 32)).all()
    with pytest.raises(ValueError, match='outside'):
        backend.crop(-1, 1, 32, 32)
    backend.close()


def test_no_validation_never_reassigns_test_rows():
    rows = pd.DataFrame({'split': ['train', 'test']})
    with pytest.raises(ValueError, match='Test rows are never reassigned'):
        split_supervised_manifest(rows, validation_fraction=.5)


def test_missing_coordinate_is_not_copied():
    rows = prepare_supervised_manifest(pd.DataFrame({'source_id': ['x'], 'target_axis': [.3]}), require_coordinates=False)
    assert np.isnan(rows.epithelial_distance_clipped_1p0.iloc[0])


def test_binary_metrics_handle_tied_scores():
    result = _binary_classification_metrics(np.array([0, 1, 0, 1]), np.repeat(.5, 4))
    assert result['binary_auroc'] == .5
    assert result['binary_average_precision'] == .5
