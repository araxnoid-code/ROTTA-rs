use std::sync::Arc;

use crate::arrayy::Arrayy;

pub struct TensorRef {
    pub value: Arc<Arrayy>,
}
