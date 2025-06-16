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
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value().matmul(&b.value());

    let tensor = Tensor::new(output);
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Matmul(a.node.clone(), b.node.clone())
    );

    tensor
}

pub fn d_matmul(a: &NodeType, b: &NodeType, grad: Arrayy) {
    // da = grad * b^t
    let d_a = grad.matmul(&b.lock().unwrap().value.clone().t());
    a.lock().as_mut().unwrap().add_grad(d_a);

    // // db = a * grad
    let d_b = a.lock().unwrap().value.clone().t().matmul(&grad);
    b.lock().as_mut().unwrap().add_grad(d_b);
}

// dot
pub fn dot(a: &Tensor, b: &Tensor) -> Tensor {
    // a dot(b) = c
    let output = a.value().dot(&b.value());

    let tensor = Tensor::new(output);
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Dot(a.node.clone(), b.node.clone())
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
fn broadcasting_tensor_non_panic(tensor_arr: &Tensor, broadcast_shape: Vec<usize>) -> Tensor {
    let arr = broadcasting(&tensor_arr.value(), broadcast_shape).unwrap_or(tensor_arr.value());

    let tensor = Tensor::new(arr);
    tensor.update_parent(vec![tensor_arr.node.clone()]);
    tensor.node.lock().unwrap().label = Some(
        BackwardLabel::Broadcasting(tensor_arr.node.clone(), tensor.value())
    );

    tensor
}

pub fn d_broadcasting_tensor(tensor_arr: &NodeType, broad_arr: Arrayy, grad: Arrayy) {
    let broadcasted_shape = &broad_arr.shape;
    let pre_shape = tensor_arr.lock().unwrap().value.shape.clone();
    let mut sum_list = vec![];

    broadcasted_shape
        .iter()
        .enumerate()
        .for_each(|(d, broad)| {
            if *broad != pre_shape[d] {
                sum_list.push(d);
            }
        });

    let mut sum = grad;

    for sum_d in sum_list {
        sum = sum_axis(&sum, sum_d);
    }

    let d_arr = reshape(&sum, pre_shape);
    tensor_arr.lock().unwrap().add_grad(d_arr);
}

// add
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let broadcast_shape = broadcast_concat(&a.value(), &b.value());

    let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
    let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

    // let broadcast_a = a;
    // let broadcast_b = b;

    let output = broadcast_a.value() + broadcast_b.value();
    let tensor = Tensor::new(output);
    tensor.update_parent(vec![broadcast_a.node.clone(), broadcast_b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Add(broadcast_a.node.clone(), broadcast_b.node.clone())
    );

    tensor
}

pub fn d_add(a: &NodeType, b: &NodeType, grad: Arrayy) {
    // f/da = 1 * grad = grad
    a.lock().as_mut().unwrap().add_grad(grad.clone());

    // f/db = 1 * grad = grad
    b.lock().as_mut().unwrap().add_grad(grad);
}

// // divided
// pub fn divided(a: &Tensor, b: &Tensor) -> Tensor {
//     let output = a.value() / b.value();

//     let tensor = Tensor::new(output);
//     tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
//     tensor.node.lock().as_mut().unwrap().label = Some(
//         BackwardLabel::Add(a.node.clone(), b.node.clone())
//     );

//     tensor
// }

// pub fn d_divided(a: &NodeType, b: &NodeType, grad: &NdArray) {
//     let epsilon = 1e-9;
//     let a_value = a.lock().unwrap().value.clone();
//     let b_value = b.lock().unwrap().value.clone();
//     // da = 1/b * grad
//     let d_a = (1.0 / &b_value) * grad;
//     a.lock().unwrap().add_grad(&d_a);

//     // db = -a/b^2 * grad
//     let d_b = -(a_value / (&b_value + epsilon).pow2()) * grad;
//     b.lock().unwrap().add_grad(&d_b);
// }
