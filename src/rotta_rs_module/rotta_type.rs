use std::sync::{ Arc, Mutex, RwLock };

use crate::rotta_rs_module::Node;

pub type NodeType = Arc<RwLock<Node>>;
