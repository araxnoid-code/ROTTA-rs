use crate::rotta_rs::{ NdArray, NodeType };

#[derive(Debug, Clone)]
pub enum BackwardLabel {
    // operation
    Matmul(NodeType, NodeType),
    Add(NodeType, NodeType),
    Diveded(NodeType, NodeType),

    // method
    Exp(NodeType, NdArray),

    // activation
    Relu(NodeType),
    Softmax(NodeType, NdArray), // node, softmax

    // loss
    SSResidual(NodeType, NodeType), // prediction, actual
    CEL(NodeType, NodeType), // prediction, actual
}
