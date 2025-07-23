use crate::{ arrayy::Arrayy, NodeType, Tensor };

// sin
pub fn sin(x: &Tensor) -> Tensor {
    Tensor::from_arrayy(x.value().sin())
}

pub fn d_sin(x: &NodeType, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    let d = x.value.cos() * grad;
    x.add_grad(d);
}

// cos
pub fn cos(x: &Tensor) -> Tensor {
    Tensor::from_arrayy(x.value().cos())
}

pub fn d_cos(x: &NodeType, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    let d = -1.0 * x.value.sin() * grad;
    x.add_grad(d);
}

// tan
pub fn tan(x: &Tensor) -> Tensor {
    Tensor::from_arrayy(x.value().tan())
}

pub fn d_tan(x: &NodeType, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    let d = x.value.tan().powi(2) * grad;
    x.add_grad(d);
}
