use crate::Tensor;

pub struct BatchNorm {
    gamma: Tensor,
    beta: Tensor,

    r_mean: Tensor,
    r_variant: Tensor,
}

impl BatchNorm {
    pub fn forward(x: &Tensor) -> Tensor {
        let shape = x.shape();
        if shape.len() <= 1 {
            panic!("error, BatchNorm input must have minimum have 2dimension");
        }

        // [N,C,W,H]
        let mut axis = vec![];
        shape
            .into_iter()
            .enumerate()
            .for_each(|(i, _)| {
                if i != 1 {
                    axis.push(i as i32);
                }
            });

        let mean = x.mean_axis_keep_dim(&axis);
        let variant = (x - &mean).powi(2).mean_axis_keep_dim(&axis);

        let eps = 1e-8;
        &(x - &mean) / &(&variant + eps).powf(0.5)
    }
}
