mod module;
pub use module::*;

mod function;
pub use function::*;

mod initialization;
pub use initialization::*;

pub trait TrainEvalHandler {
    fn train(&mut self);
    fn eval(&mut self);
}
