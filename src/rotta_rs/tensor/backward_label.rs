use crate::rotta_rs::{ NdArray, NodeType };

#[derive(Debug)]
pub enum BackwardLabel {
    // operation
    Matmul(NodeType, NodeType),
    Add(NodeType, NodeType),

    // activation
    Relu(NodeType),
    Softmax(NodeType, NdArray), // node, softmax

    // loss
    SSResidual(NodeType, NodeType), // prediction, actual
    CEL(NodeType, NodeType), // prediction, actual
}
