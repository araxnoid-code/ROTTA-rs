use rotta_rs::*;

fn main() {
    let mut model = Module::init();

    // training phase
    model.train();
    model.eval();
}
