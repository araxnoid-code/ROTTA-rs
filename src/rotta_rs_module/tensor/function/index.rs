use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor }, ShareTensor };

pub fn index(x: &Tensor, index: Vec<i32>) -> Tensor {
    let mut tensor = Tensor::from_arrayy(x.value.read().unwrap().index(index.clone()));
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Index(x.shared_tensor(), index)));

    tensor
}

pub fn index_replace(x: &Tensor, index: Vec<i32>, replace: Tensor) {
    // only can to tensor requires_gradient=false
    if !x.requires_grad() {
        x.value.write().unwrap().index_mut(index, &*replace.value.read().unwrap());
    } else {
        panic!("{}", "can't change manualy a tensor if the tensor is requires_grad=true")
    }
}

pub fn d_index(x: &ShareTensor, index: Vec<i32>, grad: &Arrayy) {
    x.grad.write().unwrap().index_mut(index, grad);
}
