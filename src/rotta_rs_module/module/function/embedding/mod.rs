mod embedding;
pub use embedding::*;

use rand::distr::Uniform;
use rand_distr::Distribution;

use crate::{ Module, Tensor };

impl Module {
    pub fn embedding_init(&mut self, vocab_num: usize, hidden: usize) -> Embedding {
        // let mut list = Vec::with_capacity(vocab_num);
        let uniform = Uniform::new(
            -(1.0 / (hidden as f64)).sqrt(),
            (1.0 / (hidden as f64)).sqrt()
        ).unwrap();

        let parameter = Tensor::from_shape_fn(vec![vocab_num, hidden], || {
            uniform.sample(&mut self.rng)
        });
        let mut parameters = self.parameters.lock().unwrap();
        parameters.push(parameter.node.clone());

        Embedding {
            num_vocab: vocab_num,
            hidden: hidden,
            parameter,
        }
    }
}
