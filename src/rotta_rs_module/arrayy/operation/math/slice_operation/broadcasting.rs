use crate::rotta_rs_module::arrayy::*;

pub fn broadcast_shape_slice(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>, String> {
    let mut shape_a = shape_a.to_vec();
    shape_a.reverse();
    let mut shape_b = shape_b.to_vec();
    shape_b.reverse();

    let shape_a_len = shape_a.len();
    let shape_b_len = shape_b.len();

    let looping = if shape_a_len > shape_b_len { shape_a_len } else { shape_b_len };

    let mut broadcasted = Vec::new();
    for i in 0..looping {
        let a = shape_a.get(i).unwrap_or(&1);
        let b = shape_b.get(i).unwrap_or(&1);

        if a == &1 {
            broadcasted.push(*b);
        } else if b == &1 {
            broadcasted.push(*a);
        } else if a == b {
            broadcasted.push(*a);
        } else {
            return Err(format!("shape {:?} and shape {:?} can't be broadcasted", shape_a, shape_b));
        }
    }
    broadcasted.reverse();

    Ok(broadcasted)
}

pub fn broadcasting_arr_slice(
    arr: (&[f32], &[usize]),
    broadcast_shape: &[usize]
) -> (Vec<f32>, Vec<usize>) {
    let (arr, shape) = arr;

    let shape_len = shape.len();
    let broadcast_shape_len = broadcast_shape.len();
    let distance = ((shape_len as f32) - (broadcast_shape_len as f32)).abs() as usize;

    if shape == broadcast_shape || shape.multiple_sum() == broadcast_shape.multiple_sum() {
        return (arr.to_vec(), broadcast_shape.to_vec());
    }

    let mut vector: Vec<f32> = Vec::with_capacity(broadcast_shape.multiple_sum());
    let mut new_shape = if shape_len == broadcast_shape_len {
        shape.to_vec()
    } else {
        let mut shape = shape.to_vec();
        for _ in 0..distance {
            shape.insert(0, 1);
        }
        shape
    };

    for d in (0..broadcast_shape_len).rev() {
        // let a_idx = (d as i32) - (distance as i32);
        // let d_a = if a_idx >= 0 { new_shape[a_idx as usize] } else { 1 };
        let d_a = new_shape[d];
        let d_b = broadcast_shape[d];
        let mut acumulate = Vec::new();

        if d_a != d_b {
            let length = (&new_shape[d..]).multiple_sum();

            let len_a = if vector.is_empty() { arr.len() } else { vector.len() };

            let count = len_a / length;

            for i in 0..count {
                let start = length * i;
                let stop = start + length;

                let slice = if vector.is_empty() {
                    &arr[start..stop]
                } else {
                    &vector[start..stop]
                };

                let copy_count = d_b;

                for _ in 0..copy_count {
                    acumulate.extend_from_slice(slice);
                }
            }

            vector = acumulate;
            new_shape[d] = d_b;
        }
    }

    (vector, new_shape)
}
