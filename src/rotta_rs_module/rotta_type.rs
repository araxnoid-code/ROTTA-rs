use std::sync::{ Arc, Mutex };

use crate::rotta_rs_module::Node;

pub type NodeType = Arc<Mutex<Node>>;
