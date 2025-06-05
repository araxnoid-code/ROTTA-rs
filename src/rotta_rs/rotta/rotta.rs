use std::sync::{ Arc, Mutex };

use crate::rotta_rs::{ Allocator, Node };

pub struct Rotta {
    allocator: Arc<Mutex<Allocator>>,
}

impl Rotta {
    pub fn init() -> Rotta {
        let allocator = Arc::new(Mutex::new(Allocator::init()));
        Rotta {
            allocator,
        }
    }
}
