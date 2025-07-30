use crate::{
    rotta_rs_module::{ arrayy::{ ArrSlice, Arrayy }, BackwardLabel, NodeType, Tensor },
    ShareTensor,
};

pub fn slice(x: &Tensor, range: &[ArrSlice]) -> Tensor {
    let mut tensor = Tensor::from_arrayy(x.value.read().unwrap().slice(range));
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Slice(x.shared_tensor(), range.to_vec())));

    tensor
}

pub fn slice_replace(x: &Tensor, range: &[ArrSlice], replace: &Tensor) {
    if !x.requires_grad() {
        // only can to tensor requires_gradient=false
        x.value.write().unwrap().slice_replace(range, &replace.value.read().unwrap());
    } else {
        panic!("{}", "can't change manualy a tensor if the tensor is requires_grad=true")
    }
}

pub fn d_slice(x: &ShareTensor, range: Vec<ArrSlice>, grad: &Arrayy) {
    x.grad.write().unwrap().slice_replace(&range, grad);
}
