// use the model
use std::{ collections::HashMap, fs::File, io::Read };

use rotta_rs::{
    softmax,
    Adam,
    ConcatTensors,
    CrossEntropyLoss,
    Embedding,
    Gru,
    LayerNorm,
    Linear,
    Module,
    Tensor,
};

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

pub struct MySeq2SeqModel {
    _model: Module,
    length: usize,
    // encoder
    embedding_encoder: Embedding,
    layer_norm_encoder: LayerNorm,
    gru_encoder: Gru,

    // decoder
    embedding_decoder: Embedding,
    layer_norm_decoder: LayerNorm,
    gru_decoder: Gru,
    linear_decoder: Linear,
}

impl MySeq2SeqModel {
    pub fn init(vocab_num: usize, hidden: usize, length: usize) -> MySeq2SeqModel {
        let mut model = Module::init();
        model.update_initialization(rotta_rs::WeightInitialization::Glorot);

        Self {
            // encoder
            embedding_encoder: model.embedding_init(vocab_num, hidden),
            layer_norm_encoder: model.layer_norm_init(&[hidden]),
            gru_encoder: model.gru_init(hidden),

            // decoder
            embedding_decoder: model.embedding_init(vocab_num, hidden),
            layer_norm_decoder: model.layer_norm_init(&[hidden]),
            gru_decoder: model.gru_init(hidden),
            linear_decoder: model.liniar_init(hidden, vocab_num),

            // model
            _model: model,
            length,
        }
    }

    pub fn encoder(&mut self, x: &Tensor) -> Option<Tensor> {
        let embedded = self.embedding_encoder.forward(&x.reshape(vec![self.length as i32]));
        let embedded = self.layer_norm_encoder.forward(&embedded);

        let mut _hidden = None;
        for i in 0..self.length {
            let x = embedded.index(vec![i as i32]).reshape(vec![1, -1]);
            let out = self.gru_encoder.forward(&x, _hidden);
            _hidden = Some(out);
        }

        _hidden
    }

    pub fn decoder(&mut self, context_vector: Option<Tensor>) -> Tensor {
        let mut x = Tensor::new([0.0]);
        let mut output = vec![];

        let mut _hidden = context_vector;
        for _ in 0..self.length {
            let embedded = self.embedding_decoder.forward(&x);
            let embedded = self.layer_norm_decoder.forward(&embedded);
            let out = self.gru_decoder.forward(&embedded, _hidden);

            let linear = self.linear_decoder.forward(&out);
            x = linear.argmax(-1);

            output.push(linear);

            _hidden = Some(out);
        }

        output.concat_tensor(0)
    }
}

fn main() {
    let mut buffer = String::new();
    let _read = File::open("./dataset/nlp/Dataset for chatbot_Georgy Silkin/dialogs.txt")
        .unwrap()
        .read_to_string(&mut buffer);

    let slicing = buffer
        .split('\n')
        .collect::<Vec<&str>>()[..5]
        .to_vec()
        .iter()
        .map(|&slice| { slice.split('\t').collect::<Vec<&str>>() })
        .collect::<Vec<Vec<&str>>>();

    let mut tokenizer = Tokenizer::init();
    tokenizer.set_up_from_slicing(&slicing);

    let length = 15;

    let mut ask_tensors = vec![];
    let mut ans_tensors = vec![];
    for ask_ans in slicing {
        let word_ask = ask_ans[0].split(' ').collect::<Vec<&str>>();
        let mut ask_index = vec![1.0;length];
        for (idx, word) in word_ask.into_iter().enumerate() {
            ask_index[idx] = *tokenizer.word2index.get(word).unwrap() as f64;
        }
        let ask_tensor = Tensor::from_vector(vec![1, length], ask_index);
        ask_tensors.push(ask_tensor);

        let word_ans = ask_ans[1].split(' ').collect::<Vec<&str>>();
        let mut ans_index = vec![1.0;length];
        for (idx, word) in word_ans.into_iter().enumerate() {
            ans_index[idx] = *tokenizer.word2index.get(word).unwrap() as f64;
        }
        let ans_tensor = Tensor::from_vector(vec![1, length], ans_index);
        ans_tensors.push(ans_tensor);
    }

    let hidden = 64;
    let mut seq2seq_model = MySeq2SeqModel::init(tokenizer.count, hidden, length);

    // loss
    let loss_fn = CrossEntropyLoss::init();

    // optimazer
    let mut optimazer = Adam::init(seq2seq_model._model.parameters(), 0.01);

    for epoch in 0..30 {
        // break;
        let mut avg = 0.0;
        for i in 0..ask_tensors.len() {
            let input = &ask_tensors[i];
            let label = &ans_tensors[i];

            // encoder
            let context_vector = seq2seq_model.encoder(input);

            // decoder
            let decoder = seq2seq_model.decoder(context_vector);

            let prediction = softmax(&decoder, -1);
            let actual = label.reshape(vec![-1]);
            let loss = loss_fn.forward(&prediction, &actual);
            avg += loss.value().value[0];

            optimazer.zero_grad();

            let backward = loss.backward();

            optimazer.optim(backward);
        }

        println!("epoch:{epoch} | loss => {}", avg / (ask_tensors.len() as f64));
    }

    // testing
    for i in 0..ask_tensors.len() {
        let input = &ask_tensors[i];
        let label = &ans_tensors[i];

        // encoder
        let context_vector = seq2seq_model.encoder(input);

        // decoder
        let decoder = seq2seq_model.decoder(context_vector);

        let prob = softmax(&decoder, -1);
        let max = prob.argmax(-1);

        // label
        let mut ask = String::new();
        for index in input.value().value {
            let mut word = tokenizer.index2word
                .get(&(index as usize))
                .unwrap()
                .clone();
            word.push(' ');
            ask.push_str(&word);
        }

        // label
        let mut actual = String::new();
        for index in label.value().value {
            let mut word = tokenizer.index2word
                .get(&(index as usize))
                .unwrap()
                .clone();
            word.push(' ');
            actual.push_str(&word);
        }

        // prediction
        let mut output = String::new();
        for index in max.value().value {
            let mut word = tokenizer.index2word
                .get(&(index as usize))
                .unwrap()
                .clone();
            word.push(' ');
            output.push_str(&word);
        }

        println!("=====================");
        println!("ask:\n{}", ask);
        println!("actual:\n{}", actual);
        println!("prediction:\n{}", output);
        println!("=====================");
    }
}
