use crate::rotta_rs::{ slice_replace_arr, ArrSlice, Arrayy, BackwardLabel, NodeType, Tensor };

pub fn slice(x: &Tensor, range: Vec<ArrSlice>) -> Tensor {
    let tensor = Tensor::from_arrayy(x.value().slice(range.clone()));
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Slice(x.node.clone(), range)));

    tensor
}

pub fn slice_replace(x: &Tensor, range: Vec<ArrSlice>, replace: &Tensor) {
    if !x.requires_grad() {
        // only can to tensor requires_gradient=false
        slice_replace_arr(&mut x.value(), range, &replace.value());
    } else {
        panic!("{}", "can't change manualy a tensor if the tensor is requires_grad=true")
    }
}

pub fn d_slice(x: &NodeType, range: Vec<ArrSlice>, grad: &Arrayy) {
    x.lock().unwrap().grad.slice_replace(range, grad);
}
