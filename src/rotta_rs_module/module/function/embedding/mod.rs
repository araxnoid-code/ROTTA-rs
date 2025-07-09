mod embedding;
pub use embedding::*;

use rand::distr::Uniform;
use rand_distr::Distribution;

use crate::{ Module, Tensor };

impl Module {
    pub fn embedding_init(&mut self, vocab_num: usize, hidden: usize) -> Embedding {
        let mut list = Vec::with_capacity(vocab_num);
        let uniform = Uniform::new(
            -(1.0 / (hidden as f64)).sqrt(),
            (1.0 / (hidden as f64)).sqrt()
        ).unwrap();

        let mut parameters = self.parameters.lock().unwrap();
        for _ in 0..vocab_num {
            let tensor = Tensor::from_shape_fn(vec![1, hidden], || uniform.sample(&mut self.rng));
            parameters.push(tensor.node.clone());
            list.push(tensor);
        }

        Embedding {
            num_vocab: vocab_num,
            hidden: hidden,
            embedding_list: list,
        }
    }
}
