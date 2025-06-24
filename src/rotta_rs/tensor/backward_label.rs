use crate::rotta_rs::{ Arrayy, NodeType };

#[derive(Debug, Clone)]
pub enum BackwardLabel {
    // operation
    Dot(NodeType, NodeType),
    Matmul(NodeType, NodeType),
    Add(NodeType, NodeType),
    Diveded(NodeType, NodeType),
    Mul(NodeType, NodeType),
    Sub(NodeType, NodeType),

    // mutation
    Index(NodeType, Vec<i32>),
    Broadcasting(NodeType, Arrayy),
    SumAxis(NodeType, i32, bool), // keep_dim
    Sum(NodeType),
    Permute(NodeType, Vec<usize>),

    // method
    Exp(NodeType, Arrayy),
    Powi(NodeType, i32),
    Ln(NodeType),
    Abs(NodeType),

    // activation
    Relu(NodeType),

    // loss
    SSResidual(NodeType, NodeType), // prediction, actual
    CEL(NodeType, NodeType), // prediction, actual
}
