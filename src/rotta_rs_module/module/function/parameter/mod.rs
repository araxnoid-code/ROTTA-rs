use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::{ arrayy::MultipleSum, Module, Tensor };

impl Module {
    pub fn add_parameter(&self, tensor: &Tensor) {
        tensor.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(tensor.shared_tensor());
    }

    pub fn init_rand_parameter(&mut self, shape: Vec<usize>) -> Tensor {
        let sum = shape.multiple_sum();
        let mut vector = vec![0.0;sum];
        for i in 0..shape.multiple_sum() {
            vector[i] = self.rng.random();
        }

        let tensor = Tensor::from_vector(shape, vector);
        tensor.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(tensor.shared_tensor());

        tensor
    }
}
