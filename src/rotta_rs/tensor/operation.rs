use std::sync::{ Arc, Mutex };

use ndarray::Axis;

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
    let grad_a = {
        let a_lock = a.lock().unwrap();
        if a_lock.value.shape() != grad.shape() {
            if a_lock.value.shape()[0] == 1 {
                let grad_sum = grad.sum_axis(Axis(0));
                let grad_shape = grad_sum.to_shape(a_lock.grad.dim()).unwrap().to_owned();
                grad_shape
            } else {
                grad.clone()
            }
        } else {
            grad.clone()
        }
    };
    a.lock().as_mut().unwrap().add_grad(&grad_a);

    // f/db = 1 * grad = grad
    let grad_b = {
        let b_lock = b.lock().unwrap();
        if b_lock.value.shape() != grad.shape() {
            if b_lock.value.shape()[0] == 1 {
                let grad_sum = grad.sum_axis(Axis(0));
                let grad_shape = grad_sum.to_shape(b_lock.grad.dim()).unwrap().to_owned();
                grad_shape
            } else {
                grad.clone()
            }
        } else {
            grad.clone()
        }
    };
    b.lock().as_mut().unwrap().add_grad(&grad_b);
}

// divided
pub fn divided(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value() / b.value();

    let tensor = Tensor::new(output);
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Add(a.node.clone(), b.node.clone())
    );

    tensor
}

pub fn d_divided(a: &NodeType, b: &NodeType, grad: &NdArray) {
    let epsilon = 1e-9;
    let a_value = a.lock().unwrap().value.clone();
    let b_value = b.lock().unwrap().value.clone();
    // da = 1/b * grad
    let d_a = (1.0 / &b_value) * grad;
    a.lock().unwrap().add_grad(&d_a);

    // db = -a/b^2 * grad
    let d_b = -(a_value / (&b_value + epsilon).pow2()) * grad;
    b.lock().unwrap().add_grad(&d_b);
}
