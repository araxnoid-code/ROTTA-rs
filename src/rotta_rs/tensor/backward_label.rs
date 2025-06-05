use crate::rotta_rs::NodeType;

#[derive(Debug)]
pub enum BackwardLabel {
    // operation
    Matmul(NodeType, NodeType),
    Add(NodeType, NodeType),

    // loss
    SSResidual(NodeType, NodeType), // prediction, actual
}
