use std::ops::Range;

use crate::Tensor;

pub struct TensorRange {
    _range: Range<usize>,
    _step: usize,
    _map: Box<dyn FnMut(f64) -> f64>,
    _shape: Vec<usize>,
}

impl TensorRange {
    pub fn init(range: Range<usize>) -> TensorRange {
        // let a = |x: f64| { x };
        TensorRange {
            _range: range,
            _step: 1,
            _map: Box::new(|x: f64| { x }),
            _shape: Vec::new(),
        }
    }

    pub fn step(self, step: usize) -> TensorRange {
        TensorRange {
            _range: self._range,
            _step: step,
            _map: self._map,
            _shape: self._shape,
        }
    }

    pub fn map<F: FnMut(f64) -> f64 + 'static>(self, f: F) -> TensorRange {
        TensorRange {
            _range: self._range,
            _step: self._step,
            _map: Box::new(f),
            _shape: self._shape,
        }
    }

    pub fn to_shape(self, shape: Vec<usize>) -> TensorRange {
        TensorRange {
            _range: self._range,
            _step: self._step,
            _map: self._map,
            _shape: shape,
        }
    }

    pub fn collect(mut self) -> Tensor {
        let mut vector = Vec::new();
        for i in self._range.step_by(self._step) {
            let x = self._map.as_mut()(i as f64);
            vector.push(x);
        }

        let shape = if self._shape.is_empty() { vec![vector.len()] } else { self._shape };

        Tensor::from_vector(shape, vector)
    }
}
