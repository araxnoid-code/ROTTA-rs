use crate::Tensor;

pub trait VectorToTensor {
    fn to_tensor(self) -> Tensor;
}

impl VectorToTensor for Vec<f64> {
    fn to_tensor(self) -> Tensor {
        Tensor::from_vector(vec![self.len()], self)
    }
}

impl VectorToTensor for Vec<f32> {
    fn to_tensor(self) -> Tensor {
        let vector = self
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<f64>>();
        Tensor::from_vector(vec![self.len()], vector)
    }
}

impl VectorToTensor for Vec<i32> {
    fn to_tensor(self) -> Tensor {
        let vector = self
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<f64>>();
        Tensor::from_vector(vec![self.len()], vector)
    }
}

impl VectorToTensor for Vec<i64> {
    fn to_tensor(self) -> Tensor {
        let vector = self
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<f64>>();
        Tensor::from_vector(vec![self.len()], vector)
    }
}
