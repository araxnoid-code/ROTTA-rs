use crate::rotta_rs::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn exp(x: &Tensor) -> Tensor {
    let exp_arr = x.value().exp();
    let tensor = Tensor::from_arrayy(exp_arr.clone());
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().unwrap().label = Some(BackwardLabel::Exp(x.node.clone(), exp_arr));

    tensor
}

pub fn d_exp(x: &NodeType, exp: &Arrayy, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    if x.requires_grad {
        let dx = exp * grad;
        x.add_grad(dx);
    }
}
