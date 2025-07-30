mod linear;

pub use linear::*;

use crate::rotta_rs_module::{ arrayy::Arrayy, Module, Tensor, WeightInitialization };

impl Module {
    pub fn liniar_init(&mut self, input: usize, output: usize) -> Linear {
        // weight
        let weight = Arrayy::arrayy_from_shape_fn(vec![input, output], || {
            match self.initialization {
                WeightInitialization::Random => self.random_initialization(),
                WeightInitialization::Glorot => self.glorot_initialization(input, output),
                WeightInitialization::He => self.he_initialization(input),
            }
        });

        let tensor_weight = Tensor::from_arrayy(weight);
        self.parameters.lock().unwrap().push(tensor_weight.shared_tensor());

        // bias
        let bias = Arrayy::arrayy_from_shape_fn(vec![1, output], || {
            match self.initialization {
                WeightInitialization::Random => self.random_initialization(),
                WeightInitialization::Glorot => self.glorot_initialization(input, output),
                WeightInitialization::He => self.he_initialization(input),
            }
        });

        let tensor_bias = Tensor::from_arrayy(bias);
        self.parameters.lock().unwrap().push(tensor_bias.shared_tensor());

        // linear cfg
        let linear_cfg = Linear {
            input,
            output,
            weight: tensor_weight,
            bias: tensor_bias,
        };

        // self.functions.push(Box::new(Arc::new(Mutex::new(linear_cfg))));

        linear_cfg
    }
}
