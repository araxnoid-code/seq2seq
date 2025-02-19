use std::{collections::HashMap, fs::File, io::Read};

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    tensor::Tensor,
};

struct Tokenizing {
    pub word_to_index: HashMap<String, i32>,
    pub index_to_word: HashMap<i32, String>,
    count: i32,
}

impl Default for Tokenizing {
    fn default() -> Self {
        let mut word_to_index = HashMap::new();
        word_to_index.insert("<SOS>".to_string(), 0);
        word_to_index.insert("<EOS>".to_string(), 1);

        let mut index_to_word = HashMap::new();
        index_to_word.insert(0, "<SOS>".to_string());
        index_to_word.insert(1, "<EOS>".to_string());

        Self {
            word_to_index,
            index_to_word,
            count: 2,
        }
    }
}

impl Tokenizing {
    fn set_up_tokenizing_sentence(&mut self, sentence: &String) {
        let words = sentence.split(" ").collect::<Vec<&str>>();
        for word in words {
            self.set_up_word_to_index(word);
        }
    }

    fn set_up_word_to_index(&mut self, word: &str) {
        if let None = self.word_to_index.get(word) {
            self.word_to_index.insert(word.to_string(), self.count);
            self.index_to_word.insert(self.count, word.to_string());
            self.count += 1;
        }
    }

    fn sentence_to_index(&self, sentence: &String) -> Vec<f32> {
        let words = sentence.split(" ").collect::<Vec<&str>>();
        let words = words
            .into_iter()
            .map(|word| self.word_to_index(word.to_string()))
            .collect::<Vec<f32>>();
        words
    }

    fn word_to_index(&self, word: String) -> f32 {
        if let Some(index) = self.word_to_index.get(&word) {
            index.clone() as f32
        } else {
            -1.0
        }
    }
}

fn main() {
    type my_backend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    // split the conversiation
    let mut raw_input = String::new();
    File::open("./data.txt")
        .unwrap()
        .read_to_string(&mut raw_input)
        .unwrap();

    let raw_input = raw_input.replace("?", "").replace(".", "").replace(",", "");
    let mut raw_inputs = raw_input.split("\n").collect::<Vec<&str>>();
    raw_inputs.pop();

    // split ask and answer
    let raw_inputs = raw_inputs
        .into_iter()
        .map(|value| {
            let ask_ans = value.split(" | ").collect::<Vec<&str>>();
            (
                ask_ans.get(0).unwrap().to_string(),
                ask_ans.get(1).unwrap().to_string(),
            )
        })
        .collect::<Vec<(String, String)>>();

    //set up tokenizing
    let mut token = Tokenizing::default();
    for (ask, ans) in &raw_inputs {
        token.set_up_tokenizing_sentence(ask);
        token.set_up_tokenizing_sentence(ans);
    }

    // tokenizing
    let mut input_tokens = vec![];
    let mut output_tokens = vec![];
    for (ask, ans) in &raw_inputs {
        let ask = token.sentence_to_index(ask);
        input_tokens.push(ask);

        let ans = token.sentence_to_index(ans);
        output_tokens.push(ans);
    }

    // tensoring data
    // let mut input_tensor = Vec::new();
    for input in input_tokens {
        // let mut sentence_tensor = Vec::new();
        let mut list = [0.0; 20];
        for (idx, word) in input.iter().enumerate() {
            // let tensor: Tensor<my_backend, 2> = Tensor::zeros([1, 20], &device);
            // tensor[idx] = word.clone();
            list[idx] = word.clone();
            // let tensor: Tensor<my_backend, 2> = Tensor::from([[word]]);
            // sentence_tensor.push(tensor);
        }
        let tensor: Tensor<my_backend, 2> = Tensor::from([list]);
        println!("{tensor}")
        // let sentence_tensor = Tensor::cat(sentence_tensor, 1);
        // input_tensor.push(sentence_tensor);
    }
    // let input_tensor = Tensor::cat(input_tensor, 0);
    // println!("{tensor}");
}
