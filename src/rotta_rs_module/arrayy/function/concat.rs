use crate::arrayy::{ Arrayy, MultipleSum };

pub fn concat_arr(arrayys: Vec<Arrayy>, dim: i32) -> Arrayy {
    let mut shape = arrayys.get(0).unwrap().shape.clone();

    let dim = if dim >= 0 {
        dim as usize
    } else {
        ((arrayys[0].shape.len() as i32) + dim) as usize
    };

    if dim >= shape.len() {
        panic!("can't concat cause dim:{dim} out of arrayy dimension:{}", shape.len());
    }

    let arrays_len = arrayys.len();
    let length = (&shape[dim..]).multiple_sum();

    shape[dim] = shape[dim] * arrays_len;

    let shape_len = shape.multiple_sum();
    let looping_count = shape_len / length;

    let mut arrayys_idx = 0;
    let mut t = 0;
    let mut output = Vec::with_capacity(shape_len);
    for _ in 0..looping_count {
        let vector = &arrayys[arrayys_idx].value[..];
        let start = t * length;
        let stop = start + length;

        let slice = &vector[start..stop];
        output.extend_from_slice(slice);

        arrayys_idx += 1;
        if arrayys_idx >= arrays_len {
            t += 1;
            arrayys_idx = 0;
        }
    }

    let arr = Arrayy::from_vector(shape, output);
    arr
}
