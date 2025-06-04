use std::sync::{ Arc, Mutex };

use crate::rotta_rs::Tensor;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value().dot(&b.value());

    let tensor = Tensor::new(output);
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);

    tensor
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value() + b.value();

    let tensor = Tensor::new(output);
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);

    tensor
}
