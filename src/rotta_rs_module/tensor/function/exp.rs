use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor }, ShareTensor };

pub fn exp(x: &Tensor) -> Tensor {
    let exp_arr = x.value.read().unwrap().exp();
    let mut tensor = Tensor::from_arrayy(exp_arr.clone());
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Exp(x.shared_tensor(), exp_arr)));

    tensor
}

pub fn d_exp(x: &ShareTensor, exp: &Arrayy, grad: &Arrayy) {
    if x.requires_grad() {
        let dx = exp * grad;
        x.add_grad(dx);
    }
}
