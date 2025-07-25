use crate::{ concat, matmul, sigmoid, tanh, Tensor };

pub struct Lstm {
    // forget parameters
    pub w_f: Tensor,
    pub b_f: Tensor,
    // input parameters
    pub w_i: Tensor,
    pub b_i: Tensor,
    pub w_c: Tensor,
    pub b_c: Tensor,
    // output parameters
    pub w_o: Tensor,
    pub b_o: Tensor,
}

#[derive(Debug)]
pub struct LSTMCellHidden {
    pub cell: Tensor,
    pub hidden: Tensor,
}

impl Lstm {
    pub fn forward(&self, x: &Tensor, lstm_cell_hidden: Option<LSTMCellHidden>) -> LSTMCellHidden {
        let (cell, hidden) = if let Some(cell_hidden) = lstm_cell_hidden {
            (cell_hidden.cell, cell_hidden.hidden)
        } else {
            let tensors = (Tensor::zeros(x.shape()), Tensor::zeros(x.shape()));
            tensors.0.set_requires_grad(false);
            tensors.1.set_requires_grad(false);
            tensors
        };

        let h_s = hidden.shape();
        let i_s = x.shape();
        if h_s != i_s {
            panic!(
                "GRU ERROR: hidden and input not same, hidden shape:{:?} input shape:{:?}",
                h_s,
                i_s
            );
        }

        let concat = concat(vec![&hidden, x], -1);

        let f = sigmoid(&(&matmul(&concat, &self.w_f) + &self.b_f));

        let i = sigmoid(&(&matmul(&concat, &self.w_i) + &self.b_i));
        let c = tanh(&(&matmul(&concat, &self.w_c) + &self.b_c));

        let new_cell = &(&cell * &f) + &(&i * &c);

        let o = sigmoid(&(&matmul(&concat, &self.w_o) + &self.b_o));
        let new_hidden = &o * &tanh(&new_cell);

        LSTMCellHidden {
            cell: new_cell,
            hidden: new_hidden,
        }
    }
}
