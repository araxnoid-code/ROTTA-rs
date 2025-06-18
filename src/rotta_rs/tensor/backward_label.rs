use crate::rotta_rs::{ Arrayy, NdArray, NodeType };

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

    // activation
    Relu(NodeType),
    Softmax(NodeType, NdArray), // node, softmax

    // loss
    SSResidual(NodeType, NodeType), // prediction, actual
    CEL(NodeType, NodeType), // prediction, actual
}
