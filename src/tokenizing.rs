use std::collections::HashMap;

pub struct Tokenizing {
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
    pub fn set_up_tokenizing_sentence(&mut self, sentence: &String) {
        let words = sentence.split(" ").collect::<Vec<&str>>();
        for word in words {
            self.set_up_word_to_index(word);
        }
    }

    pub fn set_up_word_to_index(&mut self, word: &str) {
        if let None = self.word_to_index.get(word) {
            self.word_to_index.insert(word.to_string(), self.count);
            self.index_to_word.insert(self.count, word.to_string());
            self.count += 1;
        }
    }

    pub fn sentence_to_index(&self, sentence: &String) -> Vec<f32> {
        let words = sentence.split(" ").collect::<Vec<&str>>();
        let words = words
            .into_iter()
            .map(|word| self.word_to_index(word.to_string()))
            .collect::<Vec<f32>>();
        words
    }

    pub fn word_to_index(&self, word: String) -> f32 {
        if let Some(index) = self.word_to_index.get(&word) {
            index.clone() as f32
        } else {
            -1.0
        }
    }
}
