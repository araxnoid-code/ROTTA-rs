pub trait ParDataHandler {
    type Input;
    type Output;
    fn forward(&self, data: &Self::Input) -> Self::Output;
}
