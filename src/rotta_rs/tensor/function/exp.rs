use crate::rotta_rs::{ Arrayy, BackwardLabel, NodeType, Tensor };

pub fn exp(x: &Tensor) -> Tensor {
    let exp_arr = x.value().exp();
    let tensor = Tensor::from_arrayy(exp_arr.clone());
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().unwrap().label = Some(BackwardLabel::Exp(x.node.clone(), exp_arr));

    tensor
}

pub fn d_exp(x: &NodeType, exp: &Arrayy, grad: &Arrayy) {
    let dx = exp * grad;
    x.lock().unwrap().add_grad(dx);
}
