use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn ln(x: &Tensor) -> Tensor {
    let tensor = Tensor::from_arrayy(x.value().ln());
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Ln(x.node.clone())));

    tensor
}

pub fn d_ln(x: &NodeType, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    // dx = 1/x
    if x.requires_grad {
        let dx = (1.0 / &x.value) * grad;
        x.add_grad(dx);
    }
}
