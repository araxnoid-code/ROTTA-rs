use std::sync::{ Arc, Mutex };

use ndarray::Array2;

use crate::rotta_rs::{ Node };

pub type NdArray = Array2<f64>;
pub type NodeType = Arc<Mutex<Node>>;
