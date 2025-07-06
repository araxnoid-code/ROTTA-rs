use crate::{ concat, Tensor };

pub struct Embedding {
    pub num_vocab: usize,
    pub hidden: usize,
    pub embedding_list: Vec<Tensor>,
}

impl Embedding {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let tokens = x.value().value;
        let mut map = vec![];

        for token in tokens {
            map.push(&self.embedding_list[token as usize]);
        }

        let mut shape = x.shape();
        shape.push(self.hidden);
        let concat = concat(map, 0);
        concat.node.lock().unwrap().value.shape = shape;

        concat
    }
}
