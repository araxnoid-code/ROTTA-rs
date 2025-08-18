use crate::{ arrayy::Arrayy, ShareTensor, Tensor };

// sin
pub fn sin(x: &Tensor) -> Tensor {
    Tensor::from_arrayy(x.value.read().unwrap().sin())
}

pub fn d_sin(x: &ShareTensor, grad: &Arrayy) {
    let d = x.value.read().unwrap().cos() * grad;
    x.add_grad(d);
}

// cos
pub fn cos(x: &Tensor) -> Tensor {
    Tensor::from_arrayy(x.value.read().unwrap().cos())
}

pub fn d_cos(x: &ShareTensor, grad: &Arrayy) {
    let d = -1.0 * x.value.read().unwrap().sin() * grad;
    x.add_grad(d);
}

// tan
pub fn tan(x: &Tensor) -> Tensor {
    Tensor::from_arrayy(x.value.read().unwrap().tan())
}

pub fn d_tan(x: &ShareTensor, grad: &Arrayy) {
    let d = x.value.read().unwrap().tan().powi(2) * grad;
    x.add_grad(d);
}
