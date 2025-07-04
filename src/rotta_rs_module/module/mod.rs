mod module;
pub use module::*;

mod function;
pub use function::*;

mod initialization;
pub use initialization::*;

mod batch_norm;
pub use batch_norm::*;

pub trait TrainEvalHandler {
    fn train(&mut self);
    fn eval(&mut self);
}
