//! Same finite-value percentiles and float32 normalization as production_crops.
//! Resizing and quantization remain in the reference PyTorch/NumPy backend.
use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

fn percentile(values: &mut [f32], q: f64) -> f64 {
    let index = (values.len() - 1) as f64 * q;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    let a = *values.select_nth_unstable_by(low, f32::total_cmp).1;
    let b = if low == high {
        a
    } else {
        *values.select_nth_unstable_by(high, f32::total_cmp).1
    };
    // NumPy subtracts the float32 endpoints before its float64 interpolation.
    let diff = (b - a) as f64;
    let fraction = index - low as f64;
    if fraction >= 0.5 {
        b as f64 - diff * (1.0 - fraction)
    } else {
        a as f64 + diff * fraction
    }
}

fn normalize_values(values: &mut [f32]) {
    let mut finite: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    let (mut low, mut high) = if finite.is_empty() {
        (0.0_f64, 1.0_f64)
    } else {
        (
            percentile(&mut finite, 1.0 / 100.0),
            percentile(&mut finite, 99.8 / 100.0),
        )
    };
    if !finite.is_empty() && (!(low + high).is_finite() || high <= low) {
        low = finite.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        high = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        if high <= low {
            high = low + 1.0;
        }
    }
    let low = low as f32;
    let denominator = ((high as f32) - low).max(1e-6_f32);
    for value in values {
        let normalized = (*value - low) / denominator;
        *value = if normalized.is_nan() {
            0.0
        } else {
            normalized.clamp(0.0, 1.0)
        };
    }
}

#[pyfunction]
fn normalize<'py>(
    py: Python<'py>,
    crop: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let view = crop.as_array();
    let shape = view.dim();
    // Own the input before releasing the GIL: another thread cannot mutate it.
    let mut values: Vec<f32> = view.iter().copied().collect();
    py.allow_threads(|| normalize_values(&mut values));
    let result = Array2::from_shape_vec(shape, values)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

#[pymodule]
fn spatialtrace_image_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_and_nonfinite() {
        let mut values = vec![7.0; 16];
        normalize_values(&mut values);
        assert_eq!(values, vec![0.0; 16]);
        let mut values = vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY];
        normalize_values(&mut values);
        assert_eq!(values, vec![0.0, 0.0, 1.0]);
    }
}
