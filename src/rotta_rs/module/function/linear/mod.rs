mod linear;
pub use linear::*;
use ndarray::Array2;

use crate::rotta_rs::{ Module, Tensor, WeightInitialization };

impl Module {
    pub fn liniar_init(&mut self, input: usize, output: usize) -> Linear {
        // weight
        let weight = Array2::from_shape_fn([input, output], |_| {
            match self.initialization {
                WeightInitialization::Random => self.random_initialization(),
                WeightInitialization::Glorot => self.glorot_initialization(input, output),
                WeightInitialization::He => self.he_initialization(input),
            }
        });

        let tensor_weight = Tensor::new(weight);
        self.parameters.lock().unwrap().push(tensor_weight.node.clone());

        // bias
        let bias = Array2::from_shape_fn([1, output], |_| {
            match self.initialization {
                WeightInitialization::Random => self.random_initialization(),
                WeightInitialization::Glorot => self.glorot_initialization(input, output),
                WeightInitialization::He => self.he_initialization(input),
            }
        });

        let tensor_bias = Tensor::new(bias);
        self.parameters.lock().unwrap().push(tensor_bias.node.clone());

        // linear cfg
        let linear_cfg = Linear {
            weight: tensor_weight,
            bias: tensor_bias,
        };

        linear_cfg
    }
}
