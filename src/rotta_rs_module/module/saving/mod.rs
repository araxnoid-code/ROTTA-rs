use std::fs;

use crate::{ arrayy::Arrayy, Module };
use serde::{ Serialize, Deserialize };

#[derive(Debug, Serialize, Deserialize)]
pub struct SavingParameters {
    pub params: Vec<Arrayy>,
}

impl Module {
    pub fn save(&self, path: &str) {
        let mut params = vec![];
        for shared_tensor in self.parameters.lock().unwrap().iter() {
            params.push(shared_tensor.value());
        }

        let parameter = SavingParameters {
            params,
        };

        let json = serde_json
            ::to_string(&parameter)
            .expect("SAVING ERROR: error translating parameters to save");
        drop(parameter);

        fs::write(path, json).expect("SAVING ERROR: error while writing parameter to path");
    }

    pub fn load_save(&self, path: &str) {
        let string = fs
            ::read_to_string(path)
            .expect("LOAD_SAVE ERROR: error while reading saved parameters");

        let params = serde_json
            ::from_str::<SavingParameters>(&string)
            .expect("LOAD_SAVE ERROR: error while translating saved parameters");

        if self.parameters.lock().unwrap().len() != params.params.len() {
            panic!(
                "LOAD_SAVE ERROR: the length of the parameters in the model and the parameters that have been saved are not the same"
            );
        }

        for (i, arr) in params.params.into_iter().enumerate() {
            let parameter = &self.parameters.lock().unwrap()[i];
            if parameter.value.read().unwrap().shape != arr.shape {
                panic!(
                    "LOAD_SAVE ERROR: the parameters that have been saved are not the same as the parameters in the model"
                );
            }
            parameter.update_value(arr);
        }
    }
}
