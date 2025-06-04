use std::sync::{ Arc, Mutex };

use crate::rotta_rs::{ BackwardLabel, NdArray, NodeType, Tensor };

// matmul
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value().dot(&b.value());

    let tensor = Tensor::new(output);
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Matmul(a.node.clone(), b.node.clone())
    );

    tensor
}

pub fn d_matmul(a: &NodeType, b: &NodeType, grad: &NdArray) {
    // da = b * grad
    let d_a = grad.dot(&b.lock().unwrap().value.clone().t());
    a.lock().as_mut().unwrap().add_grad(&d_a);

    // db = a * grad
    let d_b = a.lock().unwrap().value.clone().t().dot(grad);
    b.lock().as_mut().unwrap().add_grad(&d_b);
}

// add
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value() + b.value();

    let tensor = Tensor::new(output);
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Add(a.node.clone(), b.node.clone())
    );

    tensor
}

pub fn d_add(a: &NodeType, b: &NodeType, grad: &NdArray) {
    // f/da = 1 * grad = grad
    a.lock().as_mut().unwrap().add_grad(grad);

    // f/db = 1 * grad = grad
    b.lock().as_mut().unwrap().add_grad(grad);
}
