use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Lstm, LstmConfig, LstmState},
    prelude::Backend,
    tensor::{activation::softmax, Int, Tensor},
};

#[derive(Debug, Module)]
pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    lstm: Lstm<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>, state: LstmState<B, 2>) {
        let embedded = self.embedding.forward(input);
        let (x, state) = self.lstm.forward(embedded, Some(state));

        // (x, state)
    }
}

#[derive(Debug, Config)]
pub struct DecoderConfig {
    // input: usize,
    hidden: usize,
    output: usize,
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        Decoder {
            embedding: EmbeddingConfig::new(self.output, self.hidden).init(device),
            lstm: LstmConfig::new(self.hidden, self.hidden, true).init(device),
        }
    }
}
