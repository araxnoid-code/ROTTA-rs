mod random;
#[allow(unused)]
pub use random::*;

mod glorot;
#[allow(unused)]
pub use glorot::*;

mod he;
#[allow(unused)]
pub use he::*;

pub enum WeightInitialization {
    Random,
    He,
    Glorot,
}
