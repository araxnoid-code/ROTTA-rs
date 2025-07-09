use crate::rotta_rs_module::{ exp, Tensor };

#[allow(dead_code)]
pub fn sigmoid(x: &Tensor) -> Tensor {
    // 1/(1+e^-x)
    let x = 1.0 + &exp(&(-1.0 * x));
    let sigmoid = 1.0 / &x;

    sigmoid
}
