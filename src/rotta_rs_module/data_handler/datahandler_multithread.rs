pub trait DataHandlerMultiThreadTrait {
    type Input;
    type Output;
    fn forward(&self, data: &Self::Input) -> Self::Output;
}
