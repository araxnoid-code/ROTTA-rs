use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn exp(x: &Tensor) -> Tensor {
    let exp_arr = x.value().exp();
    let tensor = Tensor::from_arrayy(exp_arr.clone());
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.write().unwrap().label = Some(BackwardLabel::Exp(x.node.clone(), exp_arr));

    tensor
}

pub fn d_exp(x: &NodeType, exp: &Arrayy, grad: &Arrayy) {
    let _x = x.read().unwrap();

    if _x.requires_grad {
        let dx = exp * grad;
        x.write().unwrap().add_grad(dx);
    }
}
