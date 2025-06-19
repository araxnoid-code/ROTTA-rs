use crate::rotta_rs::{ Arrayy, NodeType };

#[derive(Debug, Clone)]
pub enum BackwardLabel {
    // operation
    Dot(NodeType, NodeType),
    Matmul(NodeType, NodeType),
    Add(NodeType, NodeType),
    Diveded(NodeType, NodeType),

    // mutation
    Broadcasting(NodeType, Arrayy),
    SumAxis(NodeType, usize, bool), // keep_dim

    // method
    Exp(NodeType, Arrayy),
    Powi(NodeType, i32),
    Ln(NodeType),

    // activation
    Relu(NodeType),

    // loss
    SSResidual(NodeType, NodeType), // prediction, actual
    CEL(NodeType, NodeType), // prediction, actual
}
