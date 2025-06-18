use crate::rotta_rs::{ Arrayy, BackwardLabel, NodeType, Tensor };

// dot
pub fn dot(a: &Tensor, b: &Tensor) -> Tensor {
    // a dot(b) = c
    let output = a.value().dot(&b.value());

    let tensor = Tensor::from_arrayy(output);
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
