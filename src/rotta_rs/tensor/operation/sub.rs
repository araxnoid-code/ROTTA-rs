use std::ops::Sub;

use crate::rotta_rs::{
    broadcast_concat,
    broadcasting_tensor_non_panic,
    Arrayy,
    BackwardLabel,
    NodeType,
    Tensor,
};

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    let broadcast_shape = broadcast_concat(&a.value(), &b.value());

    let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
    let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

    let tensor = Tensor::from_arrayy(broadcast_a.value() - broadcast_b.value());
    tensor.update_parent(vec![broadcast_a.node.clone(), broadcast_b.node.clone()]);
    tensor.update_label(
        Some(BackwardLabel::Sub(broadcast_a.node.clone(), broadcast_b.node.clone()))
    );

    tensor
}

pub fn d_sub(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    // d/da = 1  * grad = grad
    a.lock().unwrap().add_grad(grad.clone());

    // db = -1 * grad = -grad
    b.lock()
        .unwrap()
        .add_grad(-1.0 * grad);
}

// method

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        sub(self, rhs)
    }
}

impl Sub<f64> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        sub(self, &rhs)
    }
}

impl Sub<&Tensor> for f64 {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        sub(&float, rhs)
    }
}
