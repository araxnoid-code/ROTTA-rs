use crate::rotta_rs_module::{ arrayy::{ ArrSlice, Arrayy }, BackwardLabel, NodeType, Tensor };

pub fn slice(x: &Tensor, range: &[ArrSlice]) -> Tensor {
    let tensor = Tensor::from_arrayy(x.value().slice(range));
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Slice(x.node.clone(), range.to_vec())));

    tensor
}

pub fn slice_replace(x: &Tensor, range: &[ArrSlice], replace: &Tensor) {
    if !x.requires_grad() {
        // only can to tensor requires_gradient=false
        x.node.lock().unwrap().value.slice_replace(range, &replace.value());
    } else {
        panic!("{}", "can't change manualy a tensor if the tensor is requires_grad=true")
    }
}

pub fn d_slice(x: &NodeType, range: Vec<ArrSlice>, grad: &Arrayy) {
    x.lock().unwrap().grad.slice_replace(&range, grad);
}
