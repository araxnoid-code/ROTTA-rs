use std::sync::{ Arc, Mutex };

use crate::rotta_rs::Node;

pub type NodeType = Arc<Mutex<Node>>;
