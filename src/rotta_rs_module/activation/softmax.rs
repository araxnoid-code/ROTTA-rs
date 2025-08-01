use crate::rotta_rs_module::{ divided, exp, sum_axis_keep_dim, Tensor };

#[allow(dead_code)]
pub fn softmax(x: &Tensor, d: i32) -> Tensor {
    let exp = exp(x);
    let sum = sum_axis_keep_dim(&exp, &[d]);

    let s = divided(&exp, &sum);
    s
}
