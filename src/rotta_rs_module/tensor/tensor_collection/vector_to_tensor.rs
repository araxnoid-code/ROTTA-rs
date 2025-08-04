use crate::Tensor;

pub trait VectorToTensor {
    fn to_tensor(self) -> Tensor;
}

// impl VectorToTensor for Vec<f32> {
//     fn to_tensor(self) -> Tensor {
//         Tensor::from_vector(vec![self.len()], self)
//     }
// }

impl VectorToTensor for Vec<f32> {
    fn to_tensor(self) -> Tensor {
        let vector = self
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<f32>>();
        Tensor::from_vector(vec![self.len()], vector)
    }
}

impl VectorToTensor for Vec<i32> {
    fn to_tensor(self) -> Tensor {
        let vector = self
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<f32>>();
        Tensor::from_vector(vec![self.len()], vector)
    }
}

impl VectorToTensor for Vec<i64> {
    fn to_tensor(self) -> Tensor {
        let vector = self
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<f32>>();
        Tensor::from_vector(vec![self.len()], vector)
    }
}
