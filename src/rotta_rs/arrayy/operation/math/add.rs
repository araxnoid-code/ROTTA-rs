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
                .map(|v| *v + skalar)
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
