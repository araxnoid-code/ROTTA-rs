use crate::arrayy::{ Arrayy, MultipleSum };

pub fn argmin_arr(arr: &Arrayy, dim: i32) -> Arrayy {
    let shape = &arr.shape;
    let d = if dim >= 0 { dim as usize } else { ((shape.len() as i32) + dim) as usize };

    let dims_loop = (&shape[..d]).multiple_sum();
    let length = (&shape[d + 1..]).multiple_sum();
    let dim_target = shape[d];
    let ptr_vector = arr.value.as_ptr();

    let mut q = vec![];
    for i in 0..dims_loop {
        for ii in 0..length {
            let mut max: Option<(usize, f64)> = None;
            let index = ii + i * dim_target;
            for iii in 0..dim_target {
                let index = iii * length + index;
                let num = unsafe { *ptr_vector.add(index) };

                if let Some((_, num_compare)) = max {
                    if num < num_compare {
                        max = Some((iii, num));
                    }
                } else {
                    max = Some((iii, num));
                }
            }

            if let Some((idx, _)) = max {
                q.push(idx as f64);
            }
        }
    }

    let mut shape = shape.to_owned();
    shape.remove(d);
    Arrayy::from_vector(shape, q)
}
