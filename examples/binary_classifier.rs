use micrograd::{Module, Value};
use rand::distr::{Distribution, Uniform};

pub struct Neuron {
    pub w: Vec<Value>,
    pub b: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let between = Uniform::try_from(-1.0..=1.0).unwrap();
        let mut rng = rand::rng();
        let mut w = Vec::with_capacity(nin);
        w.extend(
            std::iter::repeat_with(|| between.sample(&mut rng))
                .take(nin)
                .map(Value::new),
        );
        Self {
            w,
            b: Value::new(between.sample(&mut rng)),
        }
    }

    pub fn call(&self, x: &[Value]) -> Value {
        assert!(x.len() == self.w.len());
        let value = self
            .w
            .iter()
            .zip(x)
            .fold(self.b.clone(), |acc, (w, x)| acc + (w.clone() * x.clone()));
        value.tanh()
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut res = Vec::new();
        res.extend(self.w.iter().map(|v| v.clone()));
        res.push(self.b.clone());
        res
    }
}

pub struct Layer {
    pub ws: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let mut ws = Vec::new();
        ws.extend(std::iter::repeat_with(|| Neuron::new(nin)).take(nout));
        Self { ws }
    }

    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        let mut res = Vec::new();
        res.extend(self.ws.iter().map(|neuron| neuron.call(x)));
        res
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        let mut res = Vec::new();
        for neuron in self.ws.iter() {
            res.extend(neuron.parameters())
        }
        res
    }
}

pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(mut nin: usize, nouts: &[usize]) -> Self {
        let mut layers = Vec::new();
        layers.extend(nouts.iter().map(|nout| {
            let layer = Layer::new(nin, *nout);
            nin = *nout;
            layer
        }));
        Self { layers }
    }

    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        let mut res = self.layers[0].call(x);
        if self.layers.len() > 1 {
            for layer in &self.layers[1..] {
                res = layer.call(&res);
            }
        }
        res
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        let mut res = Vec::new();
        for layer in self.layers.iter() {
            res.extend(layer.parameters())
        }
        res
    }
}

fn main() {
    let to_values = |x: [f64; 3]| -> Vec<Value> {
        let mut res = Vec::new();
        res.extend(x.into_iter().map(Value::new));
        res
    };

    let xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    let ys = [1.0, -1.0, -1.0, 1.0];
    let ys: Vec<_> = ys.into_iter().map(Value::new).collect();

    let mlp = MLP::new(3, &[4, 4, 1]);
    let mut y_preds = Vec::new();

    for i in 0..100 {
        y_preds.clear();
        for x in xs {
            let mut res = mlp.call(&(to_values(x)[..])).into_iter();
            y_preds.push(res.next().unwrap());
        }
        let loss = ys
            .iter()
            .zip(y_preds.iter())
            .fold(Value::new(0.0), |acc, (y, y_pred)| {
                acc + (y.clone() - y_pred.clone()).powi(2)
            });

        mlp.zero_grad();
        loss.backward();

        for w in mlp.parameters() {
            w.set_data(w.data() + (-0.05) * w.gradient());
        }

        println!("iteration {}: loss {:.4}", i, loss.data());
    }

    for y_pred in y_preds {
        println!("{:.4}", y_pred.data());
    }
}
