# Tensor
`ROTTA-rs` uses tensors to perform operations. Tensors are structs that have the following properties:
```rust
pub struct Tensor {
    pub id: u128,
    pub value: Arc<RwLock<Arrayy>>,
    pub grad: Arc<RwLock<Arrayy>>,
    pub parent: Vec<ShareTensor>,
    pub label: Option<BackwardLabel>,
    // requires
    pub requires_grad: Arc<RwLock<bool>>,
    pub auto_zero_grad: Arc<RwLock<bool>>,
    pub able_update_grad: Arc<RwLock<bool>>,
}
```
We can see that tensors have many properties, but this time we will focus on the properties that are most important for AI model training: `value`, `grad`, `parent`, `label`.

Let's start with `value` and `grad`.
# Value & Grad
`value` is a property that stores the value of a tensor, see the example below:
```rust
fn main() {
    let tensor = Tensor::new([1.0, 2.0, 3.0]);
}
```
The code above creates a tensor, and the values `[1.0, 2.0, 3.0]` are automatically converted to `Arrayy` and stored in the `value` property within the tensor.

`grad` is a property that stores the gradient of a tensor. By default, this property stores a value of 0 with a shape that matches the shape of the `Arrayy` value.
```rust
fn main() {
    let tensor = Tensor::new([
        [1.0, 2.0],
        [3.0, 4.0],
    ]);

    // value, dalam bentuk Arrayy
    // [[1.0, 2.0]
    //  [3.0, 4.0]]

    // grad, 
    // [[0.0, 0.0]
    //  [0.0, 0.0]]
}
```

but surely you see why in the value and grad properties, `Arrayy` is wrapped by `Arc<RwLock>`
```rust
pub struct Tensor {
    // ...
    pub value: Arc<RwLock<Arrayy>>,
    pub grad: Arc<RwLock<Arrayy>>,
    // ...
}
```
This is due to the structure of ROTTA-rs, which are interconnected using shared variables. This relationship allows backpropagation.

To better understand the concept of interconnected tensors, let's look at the `parent` and `label` properties.

# parent & label
### label
The `parent` property has a structure like this:
```rust
pub struct Tensor {
    // ...
    pub parent: Vec<ShareTensor>,
    // ...
}
```

a vector with `Shared Tensor` values, `Shared Tensor` is just an alias, the actual data type is:

```rust
pub type ShareTensor = Arc<Tensor>;
```

in other words, SharedTensor is just a Tensor wrapped by `Arc<>`, this function makes the tensor a shared variable that can be stored and changed by other `tensors`, this ability is very useful during `backrpopagation` later, before that let's discuss what happens during `forward`!

```rust
fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0],
        [3.0, 4.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0],
        [3.0, 4.0],
    ]);

    matmul(&tensor_a, &tensor_b);
}
```
The code above shows that we will perform the `matmul` operation on two tensors, `tensor_a` and `tensor_b`.

Let's take a closer look at the `matmul` function.
```rust
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value.read().unwrap().matmul(&b.value.read().unwrap());

    let mut tensor = Tensor::from_arrayy(output);

    tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Matmul(a.shared_tensor(), b.shared_tensor())));

    tensor
}
```
bisa kita lihat bagian ini:
```rust
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value.read().unwrap().matmul(&b.value.read().unwrap());

    let mut tensor = Tensor::from_arrayy(output);
    // 
}
```
The operation is performed at the `Arrayy` level on the `value` property in `tensor`, resulting in an `Arrayy` output that is immediately converted to a tensor.

Then we can see:
```rust
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // ...
    tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Matmul(a.shared_tensor(), b.shared_tensor())));

    tensor
}
```
we can see there a method to update `parent`, but what is its function?, let's look at the image below:

<div align="center">
<img width="250px" src="./../assets/matmul_neural_network.png">
</div>

As we can see above, the mathematical multiplication between two tensors, `a` and `b`, produces a new tensor, `output`.

The `output` tensor is the result of the operation between tensors `a` and `b`, causing tensors `a` and `b` to become the parents of the `output` tensor. This allows for the creation of a graph connecting all tensors from start to finish, which is necessary for backpropagation.

<div align="center">
<img width="350px" src="./../assets/graph_parents_sequence.png">
</div>

### label
Let's focus on the `label` property
```rust
pub struct Tensor {
    // ...
    pub label: Option<BackwardLabel>,
    // ...
}
```
The `label` property stores data of type `Option<BackwardLabel>`, `BackwardLabel` is an `enum` that has the structure:
```rust
pub enum BackwardLabel {
    // operation
    Dot(ShareTensor, ShareTensor),
    Matmul(ShareTensor, ShareTensor),
    Add(ShareTensor, ShareTensor),
    Diveded(ShareTensor, ShareTensor),
    Mul(ShareTensor, ShareTensor),
    Sub(ShareTensor, ShareTensor),
    // ...
}
```
The `BackwardLabel` function will store what operations have been performed and provide instructions on what operations will be performed in backpropagation. For example,

when `forwarding`:

let c = a * b;

then the variable c will store the label `Matmul(ShareTensor, ShareTensor)` along with the variables involved in the operation. In this case, it would look like `Matmul(a, b)`

in the code it will be:
```rust
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value.read().unwrap().matmul(&b.value.read().unwrap());

    let mut tensor = Tensor::from_arrayy(output);

    tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);

    // here
    tensor.update_label(Some(BackwardLabel::Matmul(a.shared_tensor(), b.shared_tensor())));
    // here

    tensor
}
```

`Note`: For initialized tensors, the label property will store the value `None` by default and can be changed with the function:
```rust
let tensor = Tensor::new([[1.0, 2.0, 3.0]]);
tensor.update_label(change here);
// be careful because this is a very crucial part of backpropagation in ROTTA-rs
```

Let's go back to this example:

let c = a * b;

When we run backpropagation with the backward method like this:

starting `backpropagation`

c.backward();

then `backpropagation` will start, the first process is to get the sequence that we have explained with the parent property, the second process is to execute the sequence from behind, the type of execution will depend on the BackwardLabel in the label property, based on the previous example that uses `matmul` then when backpropagation will be executed with the derived function of matmul, namely `d_matmul`
```rust
pub fn d_matmul(a: &ShareTensor, b: &ShareTensor, grad: &Arrayy) {
    if a.requires_grad() {
        let d_a = grad.matmul(&b.value.read().unwrap().t());
        a.add_grad(d_a);
    }

    if b.requires_grad() {
        let d_b = a.value.read().unwrap().t().matmul(grad);
        b.add_grad(d_b);
    }
}
```
The `derivative function` is used to find the `gradient` of the variable that has been executed during the forward phase. After obtaining the gradient for each variable, the grad property at each node will be added with their respective gradients.

This structure is what gives `tensor` properties a layered structure, specifically `value`, `grad`, and `SharedTensor`. This allows tensors to be connected to each other with modifiable properties.

# requires
In the tensor properties there is a `requires` section, this is the section that can be changed via the provided method, the following is an explanation of the `requires` section.
```rust
pub struct Tensor {
    // ...
    // requires
    pub requires_grad: Arc<RwLock<bool>>,
    pub auto_zero_grad: Arc<RwLock<bool>>,
    pub able_update_grad: Arc<RwLock<bool>>,
}
```

### requires_grad
```rust
fn main() {
    let tensor_a = Tensor::new([[1.0, 2.0]]);
    println!("{}", tensor_a.requires_grad()); // true
    tensor_a.set_requires_grad(false);
}
```
`requires_grad` controls the involvement of a tensor in backpropagation.

If `true` is set, the tensor will be involved in backpropagation and its gradient will be searched.

If `false` is set, the tensor will not be involved in backpropagation and its gradient will not be searched.

### auto_zero_grad
```rust
fn main() {
    let tensor_a = Tensor::new([[1.0, 2.0]]);
    println!("{}", tensor_a.auto_zero_grad()); // true
    tensor_a.set_auto_zero_grad(false);
}
```
`auto_zero_grad` is a property that sets a tensor to have its gradient removed (set to 0) after backpropagation. This property automatically runs when calling the `backward()` method on the tensor.

If `true`, the tensor will have its gradient removed after backpropagation.

If `false`, the tensor will not have its gradient removed after backpropagation.

`note`: Optimizers like `Sdg`, `RMSProp`, `Adam`, and others have a `zero_grad()` method that only removes gradients from tensors initialized within a `Module`. This is because most tensors in a `Module` have `auto_zero_grad` set to `false` for later optimization.

### able_update_grad
```rust
fn main() {
    let tensor_a = Tensor::new([[1.0, 2.0]]);
    println!("{}", tensor_a.able_update_grad()); // true
    tensor_a.set_able_update_grad(false);
}
```
`able_update_grad` is a property for a tensor that will be updated during optimization. If the tensor is not listed in the `paramters` of the `Module`, this property is useless. However, if the tensor is listed in the `parameters` of the `Module`, it will affect the tensor.

If `true`, the tensor will be updated.

If `false`, the tensor will not be updated.