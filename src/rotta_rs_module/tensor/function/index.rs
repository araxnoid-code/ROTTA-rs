use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn index(x: &Tensor, index: Vec<i32>) -> Tensor {
    let tensor = Tensor::from_arrayy(x.value().index(index.clone()));
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Index(x.node.clone(), index)));

    tensor
}

pub fn index_replace(x: &Tensor, index: Vec<i32>, replace: Tensor) {
    // only can to tensor requires_gradient=false
    if !x.requires_grad() {
        x.node.lock().unwrap().value.index_mut(index, replace.value());
    } else {
        panic!("{}", "can't change manualy a tensor if the tensor is requires_grad=true")
    }
}

pub fn d_index(x: &NodeType, index: Vec<i32>, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    x.grad.index_mut(index, grad.clone());
}
