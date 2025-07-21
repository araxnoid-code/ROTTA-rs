use std::{ collections::HashMap, fs::File, io::Read, time::SystemTime };

use rotta_rs::{ arrayy::{ argmax_arr, Arrayy }, concat, Module, Tensor };

struct Tokenizer {
    word2index: HashMap<String, usize>,
    index2word: HashMap<usize, String>,
    count: usize,
}

impl Tokenizer {
    pub fn init() -> Tokenizer {
        let mut index2word = HashMap::new();
        index2word.insert(0, "SOS".to_string());
        index2word.insert(1, "EOS".to_string());

        let mut word2index = HashMap::new();
        word2index.insert("SOS".to_string(), 0);
        word2index.insert("EOS".to_string(), 1);

        Self {
            index2word,
            word2index,
            count: 2,
        }
    }

    pub fn set_up_from_slicing(&mut self, slicing: &Vec<Vec<&str>>) {
        for ask_ans in slicing {
            for sentence in ask_ans {
                let words = sentence.split(' ').collect::<Vec<&str>>();
                for word in words {
                    if let None = self.word2index.get(word) {
                        self.word2index.insert(word.to_string(), self.count);
                        self.index2word.insert(self.count, word.to_string());
                        self.count += 1;
                    }
                }
            }
        }
    }
}

fn main() {
    let array = Tensor::rand(vec![256, 256, 512]);

    let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    array.argmax(0);

    let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    println!("{}ms", tock - tick);

    // let mut buffer = String::new();
    // let _read = File::open("./dataset/nlp/Dataset for chatbot_Georgy Silkin/dialogs.txt")
    //     .unwrap()
    //     .read_to_string(&mut buffer);

    // let slicing = buffer
    //     .split('\n')
    //     .collect::<Vec<&str>>()[..5]
    //     .to_vec()
    //     .iter()
    //     .map(|&slice| { slice.split('\t').collect::<Vec<&str>>() })
    //     .collect::<Vec<Vec<&str>>>();

    // let mut tokenizer = Tokenizer::init();
    // tokenizer.set_up_from_slicing(&slicing);

    // let length = 10;

    // let mut ask_tensors = vec![];
    // let mut ans_tensors = vec![];
    // for ask_ans in slicing {
    //     let word_ask = ask_ans[0].split(' ').collect::<Vec<&str>>();
    //     let mut ask_index = vec![0.0;length];
    //     for (idx, word) in word_ask.into_iter().enumerate() {
    //         ask_index[idx] = *tokenizer.word2index.get(word).unwrap() as f64;
    //     }
    //     let ask_tensor = Tensor::from_vector(vec![1, length], ask_index);
    //     ask_tensors.push(ask_tensor);

    //     let word_ans = ask_ans[1].split(' ').collect::<Vec<&str>>();
    //     let mut ans_index = vec![0.0;length];
    //     for (idx, word) in word_ans.into_iter().enumerate() {
    //         ans_index[idx] = *tokenizer.word2index.get(word).unwrap() as f64;
    //     }
    //     let ans_tensor = Tensor::from_vector(vec![1, length], ans_index);
    //     ans_tensors.push(ans_tensor);
    // }

    // let mut model = Module::init();
    // let embedding = model.embedding_init(tokenizer.count, 8);
    // let lstm = model.lstm_init(8);

    // for i in 0..1 {
    //     let input = &ask_tensors[i];
    //     let label = &ans_tensors[i];

    //     // encoder
    //     let cell_hidden = (|| {
    //         let embedded = embedding.forward(&input.reshape(vec![length as i32]));

    //         let mut cell_hidden = None;
    //         for i in 0..length {
    //             let x = embedded.index(vec![i as i32]).reshape(vec![1, -1]);
    //             let out = lstm.forward(&x, cell_hidden);
    //             cell_hidden = Some(out);
    //         }

    //         cell_hidden
    //     })();
    //     //

    //     // decoder
    //     //
    // }
}
