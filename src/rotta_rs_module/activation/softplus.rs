use crate::rotta_rs_module::{ exp, ln, Tensor };

pub fn softplus(x: &Tensor) -> Tensor {
    let x = 1.0 + &exp(x);
    let log = ln(&x);

    log
}
