use std::sync::{ Arc, Mutex };

use ndarray::Axis;

use crate::rotta_rs::{
    broadcast_concat,
    broadcasting,
    reshape,
    sum_axis,
    Arrayy,
    BackwardLabel,
    NdArray,
    NodeType,
    Tensor,
};

// matmul
pub fn dot(a: &Tensor, b: &Tensor) -> Tensor {
    // a * b = c
    let output = a.value().dot(b.value());

    let tensor = Tensor::new(output);
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Matmul(a.node.clone(), b.node.clone())
    );

    tensor
}

pub fn d_dot(a: &NodeType, b: &NodeType, grad: Arrayy) {
    // d/da = b * grad
    let d_a = b.lock().unwrap().value.clone() * grad.clone();
    a.lock().as_mut().unwrap().add_grad(d_a);

    // db = a * grad
    let d_b = a.lock().unwrap().value.clone() * grad.clone();
    b.lock().as_mut().unwrap().add_grad(d_b);
}

// broadcasting_tensor
fn broadcasting_tensor_non_panic(tensor_arr: &Tensor, broadcast_shape: Vec<usize>) {
    let arr = broadcasting(&tensor_arr.value(), broadcast_shape).unwrap_or(tensor_arr.value());

    let tensor = Tensor::new(arr);
    tensor.update_parent(vec![tensor_arr.node.clone()]);
    tensor.node.lock().unwrap().label = Some(
        BackwardLabel::Broadcasting(tensor_arr.node.clone(), tensor.value())
    );
}

pub fn d_broadcasting_tensor(tensor_arr: &Tensor, broad_arr: Arrayy, grad: Arrayy) {
    let broadcasted_shape = &broad_arr.shape;
    let pre_shape = tensor_arr.value().shape;
    let mut sum_list = vec![];

    broadcasted_shape
        .iter()
        .enumerate()
        .for_each(|(d, broad)| {
            if *broad != pre_shape[d] {
                sum_list.push(d);
            }
        });

    let mut sum = broad_arr * grad;
    for sum_d in sum_list {
        sum = sum_axis(&sum, sum_d);
    }

    let d_arr = reshape(&sum, pre_shape);
    tensor_arr.node.lock().unwrap().add_grad(d_arr);
}

// add
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let broadcasting_shape = broadcast_concat(&a.value(), &b.value());

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
