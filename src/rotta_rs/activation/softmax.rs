use crate::rotta_rs::{ divided, exp, sum_axis_keep_dim, Tensor };

pub fn softmax(x: &Tensor, d: usize) -> Tensor {
    let exp = exp(x);
    let sum = sum_axis_keep_dim(&exp, d);

    let s = divided(&exp, &sum);
    s
}
