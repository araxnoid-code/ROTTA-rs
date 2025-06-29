use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn abs(x: &Tensor) -> Tensor {
    let tensor = Tensor::from_arrayy(x.value().abs());
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Abs(x.node.clone())));

    tensor
}

pub fn d_abs(x: &NodeType, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    if x.requires_grad {
        let d_x = x.value.sign() * grad;
        x.add_grad(d_x);
    }
}
