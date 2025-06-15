use crate::{ Arrayy, MultipleSum };

pub fn broadcast_concat(arr_a: &Arrayy, arr_b: &Arrayy) -> Vec<usize> {
    let a_s = arr_a.shape.clone();
    let b_s = arr_b.shape.clone();

    let extended_a = {
        if a_s.len() < b_s.len() {
            let distance = b_s.len() - a_s.len();

            let shape = vec![vec![1;distance], a_s.to_vec()];
            let shape = shape.concat();
            shape.clone()
        } else {
            a_s.clone()
        }
    };

    let extended_b = {
        if b_s.len() < a_s.len() {
            let distance = a_s.len() - b_s.len();

            let shape = vec![vec![1;distance], b_s.to_vec()];
            let shape = shape.concat();
            shape.clone()
        } else {
            b_s
        }
    };

    let mut broadcasting_shape = vec![];
    for i in 0..extended_a.len() {
        if extended_a[i] >= extended_b[i] {
            broadcasting_shape.push(extended_a[i]);
        } else if extended_a[i] < extended_b[i] {
            broadcasting_shape.push(extended_b[i]);
        }
    }

    broadcasting_shape
}

pub fn broadcasting(arr: &Arrayy, broadcast_shape: Vec<usize>) -> Result<Arrayy, &'static str> {
    // can broadcasting ?
    let arr_shape = arr.shape.clone();
    let b_shape = broadcast_shape.clone();

    let extend = b_shape.len() - arr_shape.len();
    let extended_arr_shape = &vec![vec![1;extend], arr_shape.to_vec()].concat()[..];
    let mut broadcasting_target = vec![];

    for i in 0..extended_arr_shape.len() {
        if extended_arr_shape[i] == 1 || extended_arr_shape[i] == b_shape[i] {
            if extended_arr_shape[i] != b_shape[i] {
                broadcasting_target.push(i);
            }
        } else {
            return Err("can't broadcasting");
        }
    }

    if broadcasting_target.len() <= 0 {
        let mut arr = (*arr).clone();
        arr.shape = extended_arr_shape.to_vec();

        return Ok(arr);
    }

    let mut vector = arr.value.clone();
    let mut out_vec = Vec::new();

    for target in broadcasting_target {
        let mut output = vec![];
        let times = broadcast_shape[target];
        let length = (&extended_arr_shape[target..]).multiple_sum();
        let count = vector.len() / length;

        for i in 0..count {
            let start = i * length;
            let stop = start + length;
            let slice = &vector[start..stop];

            for _ in 0..times {
                output.push(slice);
            }
        }

        let concat = output.concat();
        if out_vec.len() > 0 {
            out_vec[0] = concat.clone();
        } else {
            out_vec.push(concat.clone());
        }
        vector = concat;
    }

    let output: Arrayy = Arrayy::from_vector(broadcast_shape, out_vec[0].clone());
    Ok(output)
}

pub fn matmul_broadcasting(
    arr: &Arrayy,
    broadcast_shape: Vec<usize>
) -> Result<Arrayy, &'static str> {
    // can broadcasting ?
    let arr_shape = arr.shape.clone();
    let mut b_shape = broadcast_shape.clone();

    let extend = b_shape.len() - arr_shape.len();
    let extended_arr_shape = &vec![vec![1;extend], arr_shape.to_vec()].concat()[..];
    let mut broadcasting_target = vec![];

    for i in 0..extended_arr_shape.len() - 2 {
        if extended_arr_shape[i] == 1 || extended_arr_shape[i] == b_shape[i] {
            if extended_arr_shape[i] != b_shape[i] {
                broadcasting_target.push(i);
            }
        } else {
            return Err("can't broadcasting");
        }
    }

    if broadcasting_target.len() <= 0 {
        let mut arr = (*arr).clone();
        arr.shape = extended_arr_shape.to_vec();

        return Ok(arr);
    }

    let mut vector = arr.value.clone();
    let mut out_vec = Vec::new();

    for target in broadcasting_target {
        let mut output = vec![];
        let times = broadcast_shape[target];
        let length = (&extended_arr_shape[target..]).multiple_sum();
        let count = vector.len() / length;

        for i in 0..count {
            let start = i * length;
            let stop = start + length;
            let slice = &vector[start..stop];

            for _ in 0..times {
                output.push(slice);
            }
        }

        let concat = output.concat();
        if out_vec.len() > 0 {
            out_vec[0] = concat.clone();
        } else {
            out_vec.push(concat.clone());
        }
        vector = concat;
    }

    let len = b_shape.len();
    b_shape[len - 1] = *arr_shape.last().unwrap();
    b_shape[len - 2] = arr_shape[len - 2];

    let output: Arrayy = Arrayy::from_vector(b_shape, out_vec[0].clone());
    Ok(output)
}
