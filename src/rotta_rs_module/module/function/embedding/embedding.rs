use crate::{ r, ConcatTensors, Tensor };

#[derive(Clone)]
pub struct Embedding {
    pub num_vocab: usize,
    pub hidden: usize,
    // pub embedding_list: Vec<Tensor>,
    pub parameter: Tensor,
}

impl Embedding {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let tokens = &x.value.read().unwrap().value;
        let mut map = vec![];

        for &token in tokens {
            map.push(self.parameter.slice(&[r(token as i32..(token as i32) + 1)]));
        }

        let mut shape = x.shape();
        shape.push(self.hidden);
        let concat = map.concat_tensor(0);

        concat.to_shape(shape)
    }
}
