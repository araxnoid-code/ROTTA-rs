use ndarray::{ Array2, Axis };

use crate::rotta_rs::{ BackwardLabel, NdArray, NodeType, Tensor };

pub fn softmax(x: &Tensor) -> Tensor {
    let x_value = x.node.lock().unwrap().value.clone();

    let reshape = [x_value.dim().0, 1];
    let x_exp = x_value.exp();
    let output = &x_exp / &x_exp.sum_axis(Axis(1)).to_shape(reshape).unwrap(); // sum

    let tensor = Tensor::new(output.clone());
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Softmax(x.node.clone(), output)
    );

    tensor
}

pub fn d_softmax(x: &NodeType, softmax: &NdArray, grad: &NdArray) {
    let mut vector = vec![];

    for (i, row) in softmax.iter().enumerate() {
        for (ii, coll) in softmax.iter().enumerate() {
            if i == ii {
                // row(1-coll)
                let d_softmax = row * (1.0 - coll);
                vector.push(d_softmax);
            } else {
                // -row * coll
                let d_softmax = -row * coll;
                vector.push(d_softmax);
            }
        }
    }

    let softmax_shape = softmax.shape()[softmax.shape().len() - 1];
    let shape = (softmax_shape, softmax_shape);
    let d_softmax = Array2::from_shape_vec(shape, vector).unwrap();
    let d_softmax = grad.dot(&d_softmax);
    x.lock().as_mut().unwrap().add_grad(&d_softmax);
}
