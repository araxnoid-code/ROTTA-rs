use std::ops::Mul;

use crate::rotta_rs::{
    broadcast_concat,
    broadcasting_tensor_non_panic,
    Arrayy,
    BackwardLabel,
    NodeType,
    Tensor,
};

// add
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let broadcast_shape = broadcast_concat(&a.value(), &b.value());

    let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
    let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

    let output = broadcast_a.value() * broadcast_b.value();
    let tensor = Tensor::from_arrayy(output);
    tensor.update_parent(vec![broadcast_a.node.clone(), broadcast_b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Mul(broadcast_a.node.clone(), broadcast_b.node.clone())
    );

    tensor
}

pub fn d_mul(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    // da = b * grad
    let da = &b.lock().unwrap().value * grad;
    a.lock().unwrap().add_grad(da);

    // db = a * grad
    let db = &a.lock().unwrap().value * grad;
    b.lock().unwrap().add_grad(db);
}

// method

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        mul(self, rhs)
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        mul(self, &rhs)
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        mul(&float, rhs)
    }
}
