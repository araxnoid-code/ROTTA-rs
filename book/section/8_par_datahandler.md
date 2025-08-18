# par datahandler
`par datahandler` is short for parallel datahandler is a feature in `DataHandler` that allows each sample in a `Dataset` to be executed in `parallel`.

## First, set up Dataset and DataHandler
```rust
use rotta_rs::{ DataHandler, Dataset, Tensor };

struct MyDataset {
    data: Vec<Tensor>,
    label: Vec<Tensor>,
}

impl MyDataset {
    pub fn init(data: Vec<Tensor>, label: Vec<Tensor>) -> MyDataset {
        MyDataset {
            data,
            label,
        }
    }
}

impl Dataset for MyDataset {
    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        (self.data[idx].clone(), self.label[idx].clone())
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

fn main() {
    let data_a = Tensor::new([[1.0]]);
    let label_a = Tensor::new([[2.0]]);

    let data_b = Tensor::new([[3.0]]);
    let label_b = Tensor::new([[4.0]]);

    let data_list = vec![data_a, data_b];
    let label_list = vec![label_a, label_b];
    let dataset = MyDataset::init(data_list, label_list);
    let datahandler = DataHandler::init(dataset);
}
```

## make your model AI
```rust
impl MyModel {
    fn init(model: &mut Module) -> MyModel {
        MyModel {
            linear_1: model.liniar_init(1, 8),
            linear_2: model.liniar_init(8, 1),
        }
    }
}
```

then we combine it with our previous code
```rust
use rotta_rs::{ DataHandler, Dataset, Linear, Module, Tensor };

// 
struct MyDataset {
    data: Vec<Tensor>,
    label: Vec<Tensor>,
}

impl MyDataset {
    pub fn init(data: Vec<Tensor>, label: Vec<Tensor>) -> MyDataset {
        MyDataset {
            data,
            label,
        }
    }
}

impl Dataset for MyDataset {
    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        (self.data[idx].clone(), self.label[idx].clone())
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone)]
struct MyModel {
    linear_1: Linear,
    linear_2: Linear,
    loss_fn: MSE,
}
// 

// 
impl MyModel {
    fn init(model: &mut Module) -> MyModel {
        MyModel {
            linear_1: model.liniar_init(1, 8),
            linear_2: model.liniar_init(8, 1),
        }
    }
}
// 

fn main() {
    let data_a = Tensor::new([[1.0]]);
    let label_a = Tensor::new([[2.0]]);

    let data_b = Tensor::new([[3.0]]);
    let label_b = Tensor::new([[4.0]]);

    let data_list = vec![data_a, data_b];
    let label_list = vec![label_a, label_b];
    let dataset = MyDataset::init(data_list, label_list);
    let datahandler = DataHandler::init(dataset);

    let mut model = Module::init();
    let my_model = MyModel::init(&mut model);
}
```

To be able to use `par datahandler`, we must use a trait called `ParDataHandler`
```rust
impl MyModel {
    fn init(model: &mut Module) -> MyModel {
        MyModel {
            linear_1: model.liniar_init(1, 8),
            linear_2: model.liniar_init(8, 1),
        }
    }
}

impl ParDataHandler for MyModel {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, data: &Self::Input) -> Self::Output {
        let x = self.linear_1.forward(data);
        let x = self.linear_2.forward(&x);
        x
    }
}
```

then we use the `par_by_sample` method on the DataHandler
```rust
// ...

let (loss, model) = datahandler.par_by_sample(my_model, 2, |(input, label), model| {
        let x = model.forward(input);

        let loss = model.loss_fn.forward(&x, &label);
        loss.backward();
    });
```

The `par_by_sample` method has three arguments: the first is our AI model implemented with the `ParDataHandler` trait; the second is the number of `threads` to be run; and the third is a closure containing the `forward` command, which will be executed in parallel on each sample depending on the number of `threads`.

The `par_by_sample` method will provide two outputs: a loss list and a model. The loss list contains the loss received from each thread used, and the model is useful for updating the previous model that has changed ownership due to the `move` method.

then we can add training loop and optimizer
```rust
fn main() {
    let data_a = Tensor::new([[1.0]]);
    let label_a = Tensor::new([[2.0]]);

    let data_b = Tensor::new([[3.0]]);
    let label_b = Tensor::new([[4.0]]);

    let data_list = vec![data_a, data_b];
    let label_list = vec![label_a, label_b];
    let dataset = MyDataset::init(data_list, label_list);
    let mut datahandler = DataHandler::init(dataset);

    let mut model = Module::init();
    let mut my_model = MyModel::init(&mut model);
    let mut optimazer = Adam::init(model.parameters(), 0.001);

    for epoch in 0..10 {
        let (loss, model) = datahandler.par_by_sample(my_model, 2, |(input, label), model| {
            let x = model.forward(input);
            let loss = model.loss_fn.forward(&x, label);
            loss.backward();

            loss
        });
        let loss = loss
            .iter()
            .map(|loss| loss.to_skalar())
            .sum::<f32>();
        println!("epoch:{epoch} | loss => {}", loss);
        optimazer.optim();

        my_model = model;
    }
}
```

There are several things you need to know, such as how `par_by_sample` works because it will greatly affect the output of the AI model later.

## HOW `par_by_sample` WORK?
let's look at this analogy

```sh
x = [a, b, c, d, e]
```

We have an array x containing 5 letters, each letter called a sample.

We will process each sample in a multithreaded manner using `par_by_sample`.

```sh
par_by_sample(x)
```
Suppose we use 2 threads, then `par_by_sample` will only process 2 samples which will be processed in parallel, starting from the first and second samples, namely a and b.

```sh
x = [a, b, c, d, e]
     ^  ^
    paralel
```
When finished, `par_by_sample` will update its index, we have to call `par_by_sample` again to process the next data.
```sh
par_by_sample(x)
x = [a, b, c, d, e]
           ^  ^
          paralel
```
and it continues like that...
```sh
par_by_sample(x)
x = [a, b, c, d, e]

x = [a, b, c, d, e]
     ^  ^
    paralel

par_by_sample(x)
x = [a, b, c, d, e]
                 ^
               paralel
```
When it reaches the end, `par_by_sample` will stop automatically, and when we call it again, `par_by_sample` will return to the first sample.

