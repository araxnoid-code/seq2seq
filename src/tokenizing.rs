pub struct Tokenizing {
    pub count: usize,
    pub index2word: HashMap<i32, String>,
    pub word2index: HashMap<String, i32>,
}

impl Default for Tokenizing {
    fn default() -> Self {
        let mut index2word = HashMap::new();
        index2word.insert(0, "<SOS>".to_string());
        index2word.insert(1, "<EOS>".to_string());

        let mut word2index = HashMap::new();
        word2index.insert("<SOS>".to_string(), 0);
        word2index.insert("<EOS>".to_string(), 1);

        Self {
            count: 2,
            index2word,
            word2index,
        }
    }
}

use std::{collections::HashMap, fs::File, io::Read};

impl Tokenizing {
    pub fn get_data(&mut self, location: &str) {
        let mut data = String::new();
        File::open(location)
            .unwrap()
            .read_to_string(&mut data)
            .unwrap();

        let mut data_list = data.split("\n").collect::<Vec<&str>>();
        data_list.pop();

        let data_list = data_list
            .into_iter()
            .map(|value| {
                let split = value.split(" | ").collect::<Vec<&str>>();
                (split[0].to_string(), split[1].to_string())
            })
            .collect::<Vec<(String, String)>>();

        for (ask, ans) in &data_list {
            self.set_up_sentence_to_index(ask);
            self.set_up_sentence_to_index(ans);
        }
    }

    pub fn set_up_sentence_to_index(&mut self, sentence: &str) {
        let sentence = sentence.replace(",", "").replace(".", "").replace("?", "");
        let words = sentence.split(" ").collect::<Vec<&str>>();

        for word in words {
            self.set_up_word_to_index(word);
        }
    }

    pub fn set_up_word_to_index(&mut self, word: &str) {
        if let None = self.word2index.get(word) {
            self.word2index.insert(word.to_string(), self.count as i32);
            self.index2word.insert(self.count as i32, word.to_string());
            self.count += 1;
        }
    }
}
