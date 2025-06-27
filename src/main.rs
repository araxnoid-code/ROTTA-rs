use std::{ sync::{ Arc, Mutex }, time::UNIX_EPOCH };

use crate::rotta_rs::{
    add_arr,
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
    let array = Arrayy::from_vector(vec![3, 1, 1, 1], vec![1.0, 2.0, 3.0]);
    // println!("{}", array);

    let a = broadcasting_arr_slice((array.value.as_slice(), array.shape.as_slice()), &[3, 3, 3, 2]);
    println!("{}", a);

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
