use std::ops::Sub;

use crate::rotta_rs::{
    arrayy::broadcast_concat,
    broadcasting_tensor_non_panic,
    arrayy::Arrayy,
    BackwardLabel,
    arrayy::MultipleSum,
    NodeType,
    Tensor,
};

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.value();
    let b_arr = b.value();

    if a_arr.shape.multiple_sum() == 1 || b_arr.shape.multiple_sum() == 1 {
        // skalar
        let output = a_arr - b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Sub(a.node.clone(), b.node.clone())
        );

        tensor
    } else if a_arr.shape == b_arr.shape {
        // same shape
        let output = a_arr - b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Sub(a.node.clone(), b.node.clone())
        );

        tensor
    } else {
        // broadcasting
        let broadcast_shape = broadcast_concat(&a.value(), &b.value());

        let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
        let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

        let output = broadcast_a.value() - broadcast_b.value();
        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![broadcast_a.node.clone(), broadcast_b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Sub(broadcast_a.node.clone(), broadcast_b.node.clone())
        );

        tensor
    }
}

pub fn d_sub(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    let mut a = a.lock().unwrap();
    let mut b = b.lock().unwrap();

    // d/da = 1  * grad = grad
    if a.requires_grad {
        let d_a = if a.value.shape.multiple_sum() == 1 {
            Arrayy::from_vector(a.value.shape.clone(), vec![grad.sum()])
        } else {
            grad.clone()
        };
        a.add_grad(d_a);
    }

    // db = -1 * grad = -grad
    if b.requires_grad {
        let d_b = if b.value.shape.multiple_sum() == 1 {
            Arrayy::from_vector(b.value.shape.clone(), vec![grad.sum()])
        } else {
            grad.clone()
        };

        b.add_grad(-1.0 * d_b);
    }
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
        rhs.set_requires_grad(false);
        sub(self, &rhs)
    }
}

impl Sub<&Tensor> for f64 {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        float.set_requires_grad(false);
        sub(&float, rhs)
    }
}
