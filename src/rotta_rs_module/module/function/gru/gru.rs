use crate::{ matmul, sigmoid, tanh, ConcatTensors, Tensor };

pub struct Gru {
    // reset
    pub w_r: Tensor,
    pub b_r: Tensor,
    // update
    pub w_u: Tensor,
    pub b_u: Tensor,
    // candidate
    pub w_c: Tensor,
    pub b_c: Tensor,
}

impl Gru {
    pub fn forward(&self, x: &Tensor, hidden: Option<Tensor>) -> Tensor {
        let hidden = if let Some(h) = hidden {
            h
        } else {
            let tensor = Tensor::zeros(x.shape().clone());
            tensor.set_requires_grad(false);
            tensor
        };

        let concat = vec![x, &hidden].concat_tensor(-1);

        let r = sigmoid(&(&matmul(&concat, &self.w_r) + &self.b_r));

        let u = sigmoid(&(&matmul(&concat, &self.w_u) + &self.b_u));

        let concat_x_r = vec![x, &r].concat_tensor(-1);
        let candidate = tanh(&(&matmul(&concat_x_r, &self.w_c) + &self.b_c));

        let p = &(1.0 - &u) * &candidate;

        // new hidden
        &(&hidden * &u) + &p
    }
}
