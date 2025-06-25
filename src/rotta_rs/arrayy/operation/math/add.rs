use std::{ sync::{ Arc, Mutex }, thread };

use crate::rotta_rs::*;

// function
pub fn add_arr(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let arr_a_s = arr_a.shape.clone();
    let arr_b_s = arr_b.shape.clone();

    if arr_a_s == arr_b_s {
        //
        let vector = arr_a.value
            .iter()
            .enumerate()
            .map(|(i, v)| { v + arr_b.value[i] })
            .collect::<Vec<f64>>();

        let array = Arrayy::from_vector(arr_a_s, vector);
        return array;
    } else if arr_a_s.multiple_sum() == 1 || arr_b_s.multiple_sum() == 1 {
        // skalar
        if arr_a_s.multiple_sum() == 1 {
            let skalar = arr_a.value[0];

            let vector = arr_b.value
                .iter()
                .map(|v| skalar + *v)
                .collect::<Vec<f64>>();

            Arrayy::from_vector(arr_b_s.clone(), vector)
        } else {
            let skalar = arr_b.value[0];

            let vector = arr_a.value
                .iter()
                .map(|v| *v + skalar)
                .collect::<Vec<f64>>();

            Arrayy::from_vector(arr_a_s.clone(), vector)
        }
    } else {
        // broadcasting
        let b_shape = broadcast_concat(arr_a, arr_b);

        let mut a = if let Ok(arr) = broadcasting(arr_a, b_shape.clone()) {
            arr
        } else {
            (*arr_a).clone()
        };
        let b = if let Ok(arr) = broadcasting(arr_b, b_shape) { arr } else { (*arr_b).clone() };

        for i in 0..a.value.len() {
            a.value[i] += b.value[i];
        }
        a
    }
}

pub fn add_multithread_arr(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    // multithread
    let shape_a = arr_a.shape.clone();
    let shape_b = arr_b.shape.clone();

    let mut batch = shape_a[0];
    let length = (&shape_a[1..]).multiple_sum();

    let mut stop = 0;
    let cpus = thread::available_parallelism().unwrap().get();
    let looping = if batch < cpus { batch } else { cpus };

    let new_shape = Arc::new(Mutex::new(vec![]));
    let output = Arc::new(Mutex::new(vec![vec![0.0];looping]));
    let mut handles = Vec::new();

    // broadcasting
    let (arr_a_vec, arr_b_vec, shape_a, shape_b) = if
        shape_a != shape_b &&
        shape_a.multiple_sum() != 1 &&
        shape_b.multiple_sum() != 1
    {
        // broadcasting
        let b_shape = broadcast_concat(arr_a, arr_b);

        let a = if let Ok(arr) = broadcasting(arr_a, b_shape.clone().clone()) {
            arr
        } else {
            (*arr_a).clone()
        };
        let b = if let Ok(arr) = broadcasting(arr_b, b_shape.clone()) {
            arr
        } else {
            (*arr_b).clone()
        };
        (a.value, b.value, b_shape.clone(), b_shape)
    } else {
        (arr_a.value.clone(), arr_b.value.clone(), shape_a.clone(), shape_b.clone())
    };

    for (order, i) in (0..looping).rev().enumerate() {
        let chunk = ((batch as f64) / ((i as f64) + 1.0)).ceil();
        batch -= chunk as usize;
        if chunk == 0.0 {
            break;
        }

        let start = stop;
        stop = start + (length as i32) * (chunk as i32);
        let new_shape = new_shape.clone();
        let output = output.clone();

        if shape_a == shape_b {
            let vec_a = arr_a_vec.clone()[start as usize..stop as usize].to_vec();
            let vec_b = arr_b_vec.clone()[start as usize..stop as usize].to_vec();
            let shape = shape_a.clone();
            let handle = thread::spawn(move || {
                let vector = vec_a
                    .iter()
                    .enumerate()
                    .map(|(i, v)| *v + vec_b[i])
                    .collect::<Vec<f64>>();

                *new_shape.lock().unwrap() = shape;
                output.lock().unwrap()[order] = vector;
            });
            handles.push(handle);
        } else if shape_a.multiple_sum() == 1 || shape_b.multiple_sum() == 1 {
            let shape_a = arr_a.shape.clone();
            let shape_b = arr_b.shape.clone();

            let vec_a = if shape_a.multiple_sum() == 1 {
                arr_a.value.clone()
            } else {
                arr_a.value.clone()[start as usize..stop as usize].to_vec()
            };

            let vec_b = if shape_b.multiple_sum() == 1 {
                arr_b.value.clone()
            } else {
                arr_b.value.clone()[start as usize..stop as usize].to_vec()
            };

            let handle = thread::spawn(move || {
                // skalar
                if shape_a.multiple_sum() == 1 {
                    let skalar = vec_a[0];

                    let vector = vec_b
                        .iter()
                        .map(|v| skalar + *v)
                        .collect::<Vec<f64>>();

                    *new_shape.lock().unwrap() = shape_b;
                    output.lock().unwrap()[order] = vector;
                } else {
                    let skalar = vec_b[0];

                    let vector = vec_a
                        .iter()
                        .map(|v| *v + skalar)
                        .collect::<Vec<f64>>();

                    *new_shape.lock().unwrap() = shape_a;
                    output.lock().unwrap()[order] = vector;
                }
            });
            handles.push(handle);
        }
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let vector = output.lock().unwrap().concat();
    Arrayy::from_vector(new_shape.lock().unwrap().to_owned(), vector)
}
