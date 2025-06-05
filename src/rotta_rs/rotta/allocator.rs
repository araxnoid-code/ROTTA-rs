use crate::rotta_rs::Node;

pub struct Allocator {
    pub node: Vec<Node>,
    pub free: Vec<Option<usize>>,
}

impl Allocator {
    pub fn init() -> Allocator {
        Allocator { node: Vec::new(), free: Vec::new() }
    }
}
