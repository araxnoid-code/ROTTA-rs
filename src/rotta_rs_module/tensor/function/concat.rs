use crate::{ arrayy::{ concat_arr, ArrSlice, Arrayy }, NodeType, Tensor };

pub fn concat(tensors: Vec<&Tensor>, dim: usize) -> Tensor {
    let arrayys = tensors
        .into_iter()
        .map(|tensor| tensor.value())
        .collect::<Vec<Arrayy>>();

    let tensor = Tensor::from_arrayy(concat_arr(arrayys, dim));

    tensor
}

pub fn d_concat(nodes: Vec<&NodeType>, dim: usize, pre_shape: Vec<usize>, grad: &Arrayy) {
    for node in nodes {
        let node = node.lock().unwrap();
        let shape = &node.value.shape;
        let dim_len = shape[dim];

        let slices = shape
            .into_iter()
            .enumerate()
            .map(|(i, _)| {
                if i == dim { ArrSlice(None, Some(dim_len as i32)) } else { ArrSlice(None, None) }
            })
            .collect::<Vec<ArrSlice>>();
    }
}
