use crate::{ rotta_rs_module::{ arrayy::{ ArrSlice, Arrayy } }, ShareTensor };

#[derive(Debug, Clone)]
pub enum BackwardLabel {
    // operation
    Dot(ShareTensor, ShareTensor),
    Matmul(ShareTensor, ShareTensor),
    Add(ShareTensor, ShareTensor),
    Diveded(ShareTensor, ShareTensor),
    Mul(ShareTensor, ShareTensor),
    Sub(ShareTensor, ShareTensor),

    // mutation
    Index(ShareTensor, Vec<i32>),
    Broadcasting(ShareTensor, Arrayy),
    SumAxis(ShareTensor, Vec<i32>, bool), // keep_dim
    Sum(ShareTensor),
    Permute(ShareTensor, Vec<usize>),
    Slice(ShareTensor, Vec<ArrSlice>),
    ToShape(ShareTensor, Vec<usize>),
    Concat(Vec<ShareTensor>, usize),

    // method
    Exp(ShareTensor, Arrayy),
    Powi(ShareTensor, i32),
    Powf(ShareTensor, f32),
    Ln(ShareTensor),
    Abs(ShareTensor),
    Sign(ShareTensor),
    Sin(ShareTensor),
    Cos(ShareTensor),
    Tan(ShareTensor),

    // activation
    Relu(ShareTensor),

    // loss
    SSResidual(ShareTensor, ShareTensor), // prediction, actual
    CEL(ShareTensor, ShareTensor), // prediction, actual
}
