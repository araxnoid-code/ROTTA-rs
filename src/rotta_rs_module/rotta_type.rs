use std::sync::{ Arc, Mutex, RwLock };

use crate::{ rotta_rs_module::Node, Tensor };

pub type NodeType = Arc<RwLock<Node>>;
pub type ShareTensor = Arc<RwLock<Tensor>>;
