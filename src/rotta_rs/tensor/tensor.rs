use std::{ fmt::Display, sync::{ Arc, Mutex } };

use uuid::Uuid;

use crate::rotta_rs::{ NdArray, NodeType };

#[derive(Debug)]
pub struct Tensor {
    pub node: NodeType,
}
