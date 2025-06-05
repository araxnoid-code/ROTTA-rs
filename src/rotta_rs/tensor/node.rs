use crate::rotta_rs::{ NdArray, NodeType };

#[derive(Debug)]
pub struct Node {
    pub id: usize,
    pub value: NdArray,
    pub grad: NdArray,
    pub parent: Vec<NodeType>,
}
