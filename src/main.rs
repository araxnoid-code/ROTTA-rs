use std::time::UNIX_EPOCH;

use rand_distr::num_traits::float::FloatCore;

use crate::rotta_rs::{
    dot,
    matmul,
    negative_indexing,
    permute,
    relu,
    reshape_arr,
    sigmoid,
    slice_arr,
    softmax,
    softplus,
    sum,
    sum_arr,
    sum_axis,
    sum_axis_arr,
    tanh,
    ArrSlice,
    Arrayy,
    CrossEntropyLoss,
    Module,
    SSResidual,
    Sgd,
    Tensor,
    WeightInitialization,
    MAE,
    MSE,
};

mod rotta_rs;

fn main() {
    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.00001);

    optimazer.zero_grad();
    optimazer.optim();
}
