use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Lstm, LstmConfig, LstmState},
    prelude::Backend,
    tensor::{activation::softmax, Int, Tensor},
};

#[derive(Debug, Module)]
pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    lstm: Lstm<B>,
    linear: Linear<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        state: LstmState<B, 2>,
        teaching: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 2> {
        let mut outputs = Vec::new();
        if let Some(target) = teaching.clone() {
            let input_model = vec![Tensor::from([[0]]), target.clone()];
            let input_model = Tensor::cat(input_model, 1);
            let embedded = self.embedding.forward(input_model);
            let mut state_model = state;
            for i in 0..21 {
                let embedded_slice = embedded.clone().slice([0..1, i..i + 1]);
                let (output, state) = self.lstm.forward(embedded_slice, Some(state_model));
                let output = self.linear.forward(output);
                let output = output.reshape([0, -1]);
                let output = softmax(output, 1);
                outputs.push(output);
                state_model = state;
            }
        }

        Tensor::cat(outputs, 0)
    }
}

#[derive(Debug, Config)]
pub struct DecoderConfig {
    hidden: usize,
    output: usize,
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        Decoder {
            embedding: EmbeddingConfig::new(self.output, self.hidden).init(device),
            lstm: LstmConfig::new(self.hidden, self.hidden, true).init(device),
            linear: LinearConfig::new(self.hidden, self.output).init(device),
        }
    }
}
