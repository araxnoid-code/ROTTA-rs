use std::fmt::Display;
use crate::rotta_rs::arrayy::*;

// display
impl Display for Arrayy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::new();
        self.rec_display(&self.value, &self.shape, &mut result, &"".to_string());

        f.write_str(&result)
    }
}

impl Arrayy {
    pub fn rec_display(
        &self,
        vector: &Vec<f64>,
        shape: &Vec<usize>,
        result: &mut String,
        space: &String
    ) {
        if shape.len() <= 1 {
            result.push_str(format!("{space}{:?}\n", vector).as_str());
        } else {
            result.push_str(format!("{}[\n", space).as_str());
            // println!("[");

            let d_n = shape.get(0).unwrap();
            let split_idx = (&shape[1..]).multiple_sum();
            let shape = &shape[1..].to_vec();

            let new_space = format!(" {}", space);
            for n in 0..*d_n {
                let start = n * split_idx;
                let end = start + split_idx;
                let vector = vector[start..end].to_vec();

                self.rec_display(&vector, shape, result, &new_space);
            }

            result.push_str(format!("{}]\n", space).as_str());
            // println!("]")
        }
    }
}
