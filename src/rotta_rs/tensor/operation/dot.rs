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

pub fn d_dot(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    let mut a = a.lock().unwrap();
    let mut b = b.lock().unwrap();

    // d/da = b * grad
    if a.requires_grad {
        let d_a = &b.value * grad;
        a.add_grad(d_a);
    }

    // db = a * grad
    if b.requires_grad {
        let d_b = &a.value * grad;
        b.add_grad(d_b);
    }
}
