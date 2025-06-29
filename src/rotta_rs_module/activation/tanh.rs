use crate::rotta_rs_module::{ exp, Tensor };

pub fn tanh(x: &Tensor) -> Tensor {
    // (e^x - e^-x)/(e^x + e^-x)
    let z_1 = &exp(x) - &exp(&(-1.0 * x));
    let z_2 = &exp(x) + &exp(&(-1.0 * x));
    let tanh = &z_1 / &z_2;
    tanh
}
