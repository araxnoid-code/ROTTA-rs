# Tensor
the only data types possible on tensors are f64

```rust
use rotta_rs::{ arrayy::Arrayy, * };

fn main() {
    // new()
    let tensor = Tensor::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);

    // from_vector(shape, vector)
    let tensor = Tensor::from_vector(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // from_arrayy(Arrayy)
    let arrayy = Arrayy::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);
    let tensor = Tensor::from_arrayy(arrayy);

    // from_element(shape, element)
    let tensor = Tensor::from_element(vec![2, 3], 10.0);

    // from_shape_fn(shape, FnMut -> f64)
    let tensor = Tensor::from_shape_fn(vec![2, 3], || { 1.0 });

    // arange(start, stop, step)
    let tensor = Tensor::arange(0, 10, 2);
}
```


## Basic Operations On Tensors
This version still has many shortcomings in the operations that can be performed on tensors, including:

- add
```rust
use rotta_rs::*;

fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let result = &tensor_a + &tensor_b;
    // or
    let result = add(&tensor_a, &tensor_b);
}
```

- sub
```rust
use rotta_rs::*;

fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let result = &tensor_a - &tensor_b;
    // or
    let result = sub(&tensor_a, &tensor_b);
}
```

- mul
```rust
use rotta_rs::*;

fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let result = &tensor_a * &tensor_b;
    // or
    let result = mul(&tensor_a, &tensor_b);
}
```

- divided
```rust
use rotta_rs::*;

fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let result = &tensor_a / &tensor_b;
    // or
    let result = divided(&tensor_a, &tensor_b);
}
```

- dot product
```rust
use rotta_rs::*;

fn main() {
    let tensor_a = Tensor::new([1.0, 2.0, 3.0]);

    let tensor_b = Tensor::new([1.0, 2.0, 3.0]);

    let result = dot(&tensor_a, &tensor_b);
}
```

- matmul
```rust
use rotta_rs::*;

fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ]);

    let result = matmul(&tensor_a, &tensor_b);
}
```

## Method and Function In Tensor
- index & index_replace
```rust
fn main() {
    // index
    let tensor = Tensor::new([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
        ],[
         [7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0]],
    ]);

    let a = tensor.index(vec![0, 0, 0]);
    println!("{}", a); // [1.0]

    let a = tensor.index(vec![0, 1]);
    println!("{}", a); // [4.0, 5.0, 6.0]

    let a = tensor.index(vec![-1, -1, -1]);
    println!("{}", a); // [12.0]

    // index_replace
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);
    tensor.set_requires_grad(false);

    tensor.index_replace(vec![1, 1], Tensor::new([1.0, 1.0, 1.0]));
    println!("{}", tensor)  // [
                            //  [
                            //   [1.0, 2.0, 3.0]
                            //   [4.0, 5.0, 6.0]
                            //  ]
                            //  [
                            //   [7.0, 8.0, 9.0]
                            //   [1.0, 1.0, 1.0]
                            //  ]
                            // ]


}
```

- to_shape
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.to_shape(vec![6, 2]);
    println!("{a}") // [
                    //  [1.0, 2.0]
                    //  [3.0, 4.0]
                    //  [5.0, 6.0]
                    //  [7.0, 8.0]
                    //  [9.0, 10.0]
                    //  [11.0, 12.0]
                    // ]

}
```

- reshape
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.reshape(vec![6, 2]);
    println!("{a}"); // [
                     //  [1.0, 2.0]
                     //  [3.0, 4.0]
                     //  [5.0, 6.0]
                     //  [7.0, 8.0]
                     //  [9.0, 10.0]
                     //  [11.0, 12.0]
                     // ]


    let a = tensor.reshape(vec![3, -1]);
    println!("{a}") // [
                    //  [1.0, 2.0, 3.0, 4.0]
                    //  [5.0, 6.0, 7.0, 8.0]
                    //  [9.0, 10.0, 11.0, 12.0]
                    // ]

}
```

- slice & slice_replace
```rust
fn main() {
    // slice
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.slice(vec![r(0..2), r(1..2)]);
    println!("{a}"); // [
                     //  [
                     //   [4.0, 5.0, 6.0]
                     //  ]
                     //  [
                     //   [10.0, 11.0, 12.0]
                     //  ]
                     // ]

    let a = tensor.slice(vec![r(0..2), r(1..2), r(1..-1)]);
    println!("{a}"); // [
                     //  [
                     //   [5.0, 6.0]
                     //  ]
                     //  [
                     //   [11.0, 12.0]
                     //  ]
                     // ]
    
    // slice_replace
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);
    tensor.set_requires_grad(false);

    let replace_value = Tensor::new([
        [
            [1.0, 1.0, 1.0]
        ],
        [
            [1.0, 1.0, 1.0]]
        ]);
        
    tensor.slice_replace(vec![r(0..2), r(1..2)], &replace_value);
    println!("{tensor}"); // [
                          //  [
                          //   [1.0, 2.0, 3.0]
                          //   [1.0, 1.0, 1.0]
                          //  ]
                          //  [
                          //   [7.0, 8.0, 9.0]
                          //   [1.0, 1.0, 1.0]
                          //  ]
                          // ]
    // note
    // r(0..2)  or ArrSlice(Some(0), Some(2))   => 0..2
    // r(0..)   or ArrSlice(Some(0), None)      => 0..
    // r(..2)   or ArrSlice(None, Some(2))      => ..2
    // r(..)    or ArrSlice(None, None)         => ..
}
```

- concat
```rust
fn main() {
    let tensor_a = Tensor::new([[1.0, 2.0, 3.0, 4.0, 5.0]]);
    let tensor_b = Tensor::new([[6.0, 7.0, 8.0, 9.0, 10.0]]);
    let vector = vec![&tensor_a, &tensor_b];

    let tensor = concat(vector, 0);
    println!("{}", tensor);     // [
                                //  [1.0, 2.0, 3.0, 4.0, 5.0]
                                //  [6.0, 7.0, 8.0, 9.0, 10.0]
                                // ]    
}

- permute
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.permute(vec![1, 0, 2]);
    println!("{}", a);  // [
                        //  [
                        //   [1.0, 2.0, 3.0]
                        //   [7.0, 8.0, 9.0]
                        //  ]
                        //  [
                        //   [4.0, 5.0, 6.0]
                        //   [10.0, 11.0, 12.0]
                        //  ]
                        // ]
}
```
- transpose
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.transpose((1, 2));
    println!("{}", a);  // [
                        //  [
                        //   [1.0, 4.0]
                        //   [2.0, 5.0]
                        //   [3.0, 6.0]
                        //  ]
                        //  [
                        //   [7.0, 10.0]
                        //   [8.0, 11.0]
                        //   [9.0, 12.0]
                        //  ]
                        // ]


    let a = tensor.transpose((-3, -1));
    println!("{}", a);  // [
                        //  [
                        //   [1.0, 7.0]
                        //   [4.0, 10.0]
                        //  ]
                        //  [
                        //   [2.0, 8.0]
                        //   [5.0, 11.0]
                        //  ]
                        //  [
                        //   [3.0, 9.0]
                        //   [6.0, 12.0]
                        //  ]
                        // ]


    let a = tensor.t(); // same with tensor.transpose((-1, -2));
    println!("{}", a);  // [
                        //  [
                        //   [1.0, 4.0]
                        //   [2.0, 5.0]
                        //   [3.0, 6.0]
                        //  ]
                        //  [
                        //   [7.0, 10.0]
                        //   [8.0, 11.0]
                        //   [9.0, 12.0]
                        //  ]
                        // ]

}
```

- sum
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.sum();
    println!("{}", a); // [78.0]
}
```

- sum_axis
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.sum_axis(&[1]);
    println!("{}", a);  // [
                        //  [5.0, 7.0, 9.0]
                        //  [17.0, 19.0, 21.0]
                        // ]


    let a = tensor.sum_axis(&[-1]);
    println!("{}", a);  // [
                        //  [6.0, 15.0]
                        //  [24.0, 33.0]
                        // ]

    let a = tensor.sum_axis_keep_dim(&[-1]);
    println!("{}", a);  // [
                        //  [
                        //   [6.0]
                        //   [15.0]
                        //  ]
                        //  [
                        //   [24.0]
                        //   [33.0]
                        //  ]
                        // ]
    
    let a = tensor.sum_axis_keep_dim(&[0, 2]);
    println!("{a}")     // [
                        //  [
                        //   [30.0]
                        //   [48.0]
                        //  ]
                        // ]


}
```

- mean
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.mean();
    println!("{a}"); // [6.5]
}
```

- mean axis
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.mean_axis(&[0]);
    println!("{a}");    // [
                        //  [4.0, 5.0, 6.0]
                        //  [7.0, 8.0, 9.0]
                        // ]


    let a = tensor.mean_axis_keep_dim(&[-1]);
    println!("{a}");    // [
                        //  [
                        //   [2.0]
                        //   [5.0]
                        //  ]
                        //  [
                        //   [8.0]
                        //   [11.0]
                        //  ]
                        // ]



    let a = tensor.mean_axis_keep_dim(&[0, -1]);
    println!("{a}")     // [
                        //  [
                        //   [5.0]
                        //   [8.0]
                        //  ]
                        // ]

}
```

- other functions
```rust
fn main() {
    let tensor = Tensor::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);

    let absolut = tensor.abs();
    let power = tensor.powi(2);
    let power = tensor.powf(2);
    let natural_log = tensor.ln();
    let exp = tensor.exp();
    let sign = tensor.sign();
}
```