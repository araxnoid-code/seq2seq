use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use burn::tensor::activation::{sigmoid, softplus};
use burn::tensor::loss;
use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::{MseLoss, Reduction},
        Linear, LinearConfig,
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor},
};

#[derive(Module, Debug)]
struct Model<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input.forward(input);
        let x = softplus(x, 1.0);
        let x = self.output.forward(x);
        x
    }
}

#[derive(Debug, Config)]
struct ModelConfig {
    input: usize,
    hidden: usize,
    output: usize,
}

impl ModelConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            input: LinearConfig::new(self.input, self.hidden).init(device),
            hidden: LinearConfig::new(self.hidden, self.hidden).init(device),
            output: LinearConfig::new(self.hidden, self.output).init(device),
        }
    }
}

fn main() {
    // type MyBackend = Wgpu<f32, i32>;
    println!("Hello, world!");
    run::<Autodiff<Wgpu>>(&WgpuDevice::default());
}

#[derive(Config)]
struct TrainingConfig {
    #[config(default = 400)]
    num_epoch: usize,

    #[config(default = 42)]
    seed: u64,

    #[config(default = 0.1)]
    lr: f64,

    //
    model: ModelConfig,
    optimazer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: &B::Device) {
    let config_model = ModelConfig::new(1, 2, 1);
    let config_optimazer = AdamConfig::new();
    let config = TrainingConfig::new(config_model, config_optimazer);
    B::seed(config.seed);

    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimazer.init();

    let input: Tensor<B, 2> = Tensor::from([[0.0], [0.5], [1.0]]);
    let observed: Tensor<B, 2> = Tensor::from([[0.0], [1.0], [0.0]]);

    for epoch in 0..config.num_epoch {
        let pred = model.forward(input.clone());
        // let loss = MseLoss::new().forward(pred.clone(), observed.clone(), Reduction::Auto);

        let loss_2 = observed.clone() - pred.clone();
        let loss_2 = loss_2.clone() * loss_2;
        let loss = loss_2.sum();

        println!("{epoch}: loss => {:?}", loss.clone().into_scalar());

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        model = optim.step(config.lr, model, grads);
    }

    model.valid();
    let prediction = model.forward(input);
    println!("{}", prediction);
}
