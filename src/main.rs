use std::{ sync::{ Arc, Mutex }, time::UNIX_EPOCH };

use crate::rotta_rs::{
    add_arr,
    broadcast_concat,
    broadcast_shape_slice,
    broadcasting,
    broadcasting_arr_slice,
    matmul_2d,
    matmul_2d_slice,
    matmul_nd,
    matmul_nd_slice,
    par_add_arr,
    Arrayy,
};

mod rotta_rs;

fn main() {
    let a = Arrayy::from_vector(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Arrayy::from_vector(vec![2, 1, 1, 1], vec![5.0, 6.0]);

    let shape = broadcast_concat(&a, &b);

    // println!("{:?}", shape);

    let a = broadcasting(&a, shape.clone()).unwrap();
    let b = broadcasting(&b, shape.clone()).unwrap();

    // println!("{}", a);
    // println!("{}", b);

    // let data = 5012;
    // // println!("{}", data);
    // let x = Arrayy::ones(vec![data, data]);
    // let z = Arrayy::ones(vec![data, data]);

    // let mut avg = 0;
    // for _ in 0..20 {
    //     let tik = std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    //     // broadcasting(&x, vec![2, data, data]).unwrap();
    //     broadcasting_arr_slice((x.value.as_slice(), x.shape.as_slice()), &[2, data, data]);
    //     // add_arr(&x, &z);
    //     // add_multithread_arr(&x, &z);

    //     let tok = std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    //     println!("{}ms", tok - tik);
    //     avg += tok - tik;
    // }
    // println!("avg multi {}ms", avg / 10)
}
