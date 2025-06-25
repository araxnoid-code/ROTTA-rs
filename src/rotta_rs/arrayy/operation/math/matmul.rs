use std::{ sync::{ Arc, Mutex }, thread };

use crate::rotta_rs::*;

// function
pub fn matmul_2d(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let arr_a_s = arr_a.shape.clone();
    let arr_b_s = arr_b.shape.clone();
    if arr_a_s.len() == 2 && arr_b_s.len() == 2 {
        // 2x2 * 2x2
        let m = arr_a.shape[arr_a.shape.len() - 2];
        let o = arr_b.shape.last().unwrap();
        let n = arr_a.shape.last().unwrap();

        let mut vector = vec![];
        for row in 0..m {
            for coll in 0..*o {
                let mut sum = 0.0;
                for i in 0..*n {
                    sum =
                        sum +
                        (
                            arr_a.index(vec![row as i32, i as i32]) *
                            arr_b.index(vec![i as i32, coll as i32])
                        ).value[0];
                }
                vector.push(sum);
            }
        }

        let array = Arrayy::from_vector(vec![m, *o], vector);

        return array;
    } else {
        panic!("error can't matmul")
    }
}

pub fn matmul_nd(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    if arr_a.shape.len() == 1 || arr_b.shape.len() == 1 {
        panic!("can't matmul array cause, {:?} will matmul with {:?}", arr_a.shape, arr_b.shape);
    }

    //
    if arr_a.shape.len() == 2 && arr_b.shape.len() == 2 {
        matmul_2d(arr_a, arr_b)
    } else {
        let shape_a = arr_a.shape.clone();
        let shape_b = arr_b.shape.clone();
        let mut index = vec![];
        let mut d = 0;

        let mut shape = shape_a.clone();
        let len = shape.len();
        shape[len - 1] = *shape_b.last().unwrap();
        shape[len - 2] = shape_a[shape_a.len() - 2];
        let mut vector = Vec::with_capacity(shape.as_slice().multiple_sum());

        //

        loop {
            if (shape_a.len() as i32) - (d as i32) == 2 {
                // matmul operation
                let slice_range = index
                    .iter()
                    .map(|d| { ArrSlice(Some(*d), Some(d + 1)) })
                    .collect::<Vec<ArrSlice>>();

                let slice_a = slice_arr(&arr_a, slice_range.clone()).squeeze();
                let slice_b = slice_arr(&arr_b, slice_range).squeeze();

                let matmul = matmul_2d(&slice_a, &slice_b);
                vector.extend(matmul.value.into_iter());

                d -= 1;
            } else {
                if let None = index.get(d) {
                    index.push(0);
                    d += 1;
                } else {
                    index[d] += 1;

                    if index[d] >= (shape_a[d] as i32) {
                        index.pop();

                        if d == 0 {
                            break;
                        } else {
                            d -= 1;
                        }
                    } else {
                        d += 1;
                    }
                }
            }
        }

        Arrayy::from_vector(shape, vector)
    }
}

pub fn matmul_multithread(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    // multithread
    let shape_a = arr_a.shape.clone();
    let shape_b = arr_b.shape.clone();

    let mut batch = shape_a[0];
    let length = (&shape_a[1..]).multiple_sum();

    let mut stop = 0;
    let cpus = thread::available_parallelism().unwrap().get();
    let looping = if batch < cpus { batch } else { cpus };
    let output = Arc::new(Mutex::new(vec![vec![0.0];looping]));

    let mut handles = Vec::new();
    for (order, i) in (0..looping).rev().enumerate() {
        let chunk = ((batch as f64) / ((i as f64) + 1.0)).ceil();
        batch -= chunk as usize;
        if chunk == 0.0 {
            break;
        }

        let start = stop;
        stop = start + (length as i32) * (chunk as i32);
        let vec = arr_a.value[start as usize..stop as usize].to_vec();
        let mut shape = shape_a.clone();
        shape[0] = chunk as usize;

        let output = output.clone();
        let arr_b = arr_b.clone();
        let handle = thread::spawn(move || {
            let arr = Arrayy::from_vector(shape, vec);

            let result = arr.matmul(&arr_b);

            output.lock().unwrap()[order] = result.value;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let mut shape = shape_a.clone();
    let len = shape.len();
    shape[len - 1] = *shape_b.last().unwrap();
    shape[len - 2] = shape_a[shape_a.len() - 2];

    let vector = output.lock().unwrap().concat();

    Arrayy::from_vector(shape, vector)
}
