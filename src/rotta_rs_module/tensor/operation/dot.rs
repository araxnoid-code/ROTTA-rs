use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

// dot
pub fn dot(a: &Tensor, b: &Tensor) -> Tensor {
    // a dot(b) = c
    let output = a.value().dot(&b.value());

    let tensor = Tensor::from_arrayy(output);
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.write().unwrap().label = Some(BackwardLabel::Dot(a.node.clone(), b.node.clone()));

    tensor
}

pub fn d_dot(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    let mut _a = a.write().unwrap();
    let mut _b = b.write().unwrap();

    // d/da = b * grad
    if _a.requires_grad {
        let d_a = &_b.value * grad;
        _a.add_grad(d_a);
    }

    // db = a * grad
    if _b.requires_grad {
        let d_b = &_a.value * grad;
        _b.add_grad(d_b);
    }
}
