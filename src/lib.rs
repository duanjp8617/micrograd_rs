use std::cell::RefCell;
use std::collections::HashSet;
use std::convert::From;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::rc::Rc;

use rand::Rng;

type GradFunc = Box<dyn Fn(f64)>;
type ObjPtr = Rc<RefCell<ValueInner>>;

struct ValueInner {
    data: f64,
    gradient: f64,
    prevs: Vec<ObjPtr>,
    grad_func: GradFunc,
}

impl ValueInner {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            gradient: 0.0,
            prevs: vec![],
            grad_func: Box::new(|_| {}),
        }
    }
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner::new(data))))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn set_data(&self, value: f64) {
        self.0.borrow_mut().data = value;
    }

    pub fn gradient(&self) -> f64 {
        self.0.borrow().gradient
    }

    pub fn reset_grad(&self) {
        self.0.borrow_mut().gradient = 0.0;
    }

    pub fn backward(&self) {
        fn dfs(curr: ObjPtr, set: &mut HashSet<usize>, reverse_top: &mut Vec<ObjPtr>) {
            for n in curr.borrow().prevs.iter() {
                let node_idx = n.as_ptr() as usize;
                if !set.contains(&node_idx) {
                    set.insert(node_idx);
                    dfs(n.clone(), set, reverse_top);
                }
            }
            reverse_top.push(curr.clone());
        }

        let mut set = HashSet::new();
        let mut reverse_top = Vec::new();
        let curr = self.0.clone();
        set.insert(curr.as_ptr() as usize);
        dfs(curr, &mut set, &mut reverse_top);

        reverse_top.last().unwrap().borrow_mut().gradient = 1.0;
        for node in reverse_top.iter().rev() {
            let node = node.borrow();
            (node.grad_func)(node.gradient);
        }
    }

    pub fn powf(self, exponent: f64) -> Value {
        let x = self.0.clone();
        // d(x^exponent)/dx -> exponent * (x)^(exponent-1) * parent_grad
        let grad_func = Box::new(move |parent_grad: f64| {
            let result = exponent * (x.borrow().data.powf(exponent - 1.0)) * parent_grad;
            x.borrow_mut().gradient += result;
        });

        let val = ValueInner {
            data: self.data().powf(exponent),
            gradient: 0.0,
            prevs: vec![self.0.clone()],
            grad_func,
        };

        Value(Rc::new(RefCell::new(val)))
    }

    pub fn powi(self, exponent: i32) -> Value {
        self.powf(exponent.into())
    }

    pub fn relu(self) -> Value {
        let x = self.0.clone();
        let grad_func = Box::new(move |parent_grad: f64| {
            let d = if x.borrow().data <= 0.0 { 0.0 } else { 1.0 };
            x.borrow_mut().gradient += d * parent_grad
        });

        let val = ValueInner {
            data: if self.data() <= 0.0 { 0.0 } else { self.data() },
            gradient: 0.0,
            prevs: vec![self.0.clone()],
            grad_func,
        };

        Value(Rc::new(RefCell::new(val)))
    }

    pub fn exp(self) -> Value {
        let exp_res = self.data().exp();

        let x = self.0.clone();
        // d(e^x)/dx -> e^x * parent_grad;
        let grad_func =
            Box::new(move |parent_grad: f64| x.borrow_mut().gradient += exp_res * parent_grad);

        let val = ValueInner {
            data: exp_res,
            gradient: 0.0,
            prevs: vec![self.0.clone()],
            grad_func,
        };

        Value(Rc::new(RefCell::new(val)))
    }

    pub fn _tanh(self) -> Value {
        let e2x = (2.0 * self.data()).exp();
        let tanh_res = (e2x - 1.0) / (e2x + 1.0);

        let x = self.0.clone();
        // d(tanh(x))/dx -> (1-tanh(x)^2) * parent_grad;
        let grad_func = Box::new(move |parent_grad: f64| {
            x.borrow_mut().gradient += (1.0 - tanh_res.powi(2)) * parent_grad
        });

        let val = ValueInner {
            data: tanh_res,
            gradient: 0.0,
            prevs: vec![self.0.clone()],
            grad_func,
        };

        Value(Rc::new(RefCell::new(val)))
    }

    pub fn tanh(self) -> Value {
        let e2x = (Value::from(2.0) * self.clone()).exp();
        (e2x.clone() - 1.0.into()) / (e2x + 1.0.into())
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Self::new(value.into())
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        let x = self.0.clone();
        let y = rhs.0.clone();
        // d(x + y)/dx -> 1 * parent_grad
        // d(x + y)/dy -> 1 * parent_grad
        let grad_func = Box::new(move |parent_grad: f64| {
            x.borrow_mut().gradient += parent_grad;
            y.borrow_mut().gradient += parent_grad;
        });

        let val = ValueInner {
            data: self.data() + rhs.data(),
            gradient: 0.0,
            prevs: vec![self.0.clone(), rhs.0.clone()],
            grad_func,
        };

        Value(Rc::new(RefCell::new(val)))
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        let x = self.0.clone();
        // d(-x)/dx -> -1 * parent_grad
        let grad_func = Box::new(move |parent_grad: f64| x.borrow_mut().gradient -= parent_grad);

        let val = ValueInner {
            data: -self.data(),
            gradient: 0.0,
            prevs: vec![self.0.clone()],
            grad_func,
        };

        Value(Rc::new(RefCell::new(val)))
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        self + (-rhs)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        let x = self.0.clone();
        let y = rhs.0.clone();
        // d(x * y)/dx -> y * parent_grad
        // d(x * y)/dy -> x * parent_grad
        let grad_func = Box::new(move |parent_grad: f64| {
            x.borrow_mut().gradient += y.borrow().data * parent_grad;
            y.borrow_mut().gradient += x.borrow().data * parent_grad;
        });

        let val = ValueInner {
            data: self.data() * rhs.data(),
            gradient: 0.0,
            prevs: vec![self.0.clone(), rhs.0.clone()],
            grad_func,
        };

        Value(Rc::new(RefCell::new(val)))
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Value {
        self * (rhs.powi(-1))
    }
}

impl AddAssign for Value {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl SubAssign for Value {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl MulAssign for Value {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs
    }
}

impl DivAssign for Value {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs
    }
}

pub trait Module {
    fn zero_grad(&self) {
        for value in self.parameters() {
            value.reset_grad();
        }
    }

    fn parameters(&self) -> Vec<Value>;
}

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::rng();
        let mut w = Vec::with_capacity(nin);
        w.extend(
            std::iter::repeat_with(|| rng.random_range(-1.0..=1.0))
                .take(nin)
                .map(Value::new),
        );
        Self {
            w,
            b: Value::new(rng.random_range(-1.0..=1.0)),
        }
    }

    pub fn call(&self, x: &[Value]) -> Value {
        assert!(x.len() == self.w.len());
        self.w
            .iter()
            .zip(x)
            .fold(self.b.clone(), |acc, (w, x)| acc + (w.clone() * x.clone()))
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
    ws: Vec<Neuron>,
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
    layers: Vec<Layer>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_correct() {
        let compute_graph = || -> f64 {
            let a = Value::new(-4.0);
            let b = Value::new(2.0);
            let mut c = a.clone() + b.clone();
            let mut d = a.clone() * b.clone() + b.clone().powi(3);
            c += c.clone() + 1.into();
            c += Value::from(1) + c.clone() + (-a.clone());
            d += d.clone() * 2.into() + (b.clone() + a.clone()).relu();
            d += Value::from(3) * d.clone() + (b.clone() - a.clone()).relu();
            let e = c.clone() - d.clone();
            let f = e.clone().powi(2);
            let mut g = f.clone() / 2.0.into();
            g += Value::from(10.0) / f.clone();

            g.data()
        };

        let rust_native = || -> f64 {
            fn relu(n: f64) -> f64 {
                if n <= 0.0 {
                    0.0
                } else {
                    n
                }
            }

            let a: f64 = -4.0;
            let b: f64 = 2.0;
            let mut c = a + b;
            let mut d = a * b + b.powi(3);
            c += c + 1.0;
            c += 1.0 + c + (-a);
            d += d * 2.0 + relu(b + a);
            d += 3.0 * d + relu(b - a);
            let e = c - d;
            let f = e.powi(2);
            let mut g = f / 2.0;
            g += 10.0 / f;
            g
        };

        assert_eq!(format!("{:.4}", compute_graph()), "24.7041");
        assert_eq!(compute_graph(), rust_native());
    }

    #[test]
    fn backward_test() {
        let compute_graph = || -> (f64, f64) {
            let a = Value::new(-4.0);
            let b = Value::new(2.0);
            let mut c = a.clone() + b.clone();
            let mut d = a.clone() * b.clone() + b.clone().powi(3);
            c += c.clone() + 1.into();
            c += Value::from(1) + c.clone() + (-a.clone());
            d += d.clone() * 2.into() + (b.clone() + a.clone()).relu();
            d += Value::from(3) * d.clone() + (b.clone() - a.clone()).relu();
            let e = c.clone() - d.clone();
            let f = e.clone().powi(2);
            let mut g = f.clone() / 2.0.into();
            g += Value::from(10.0) / f.clone();

            g.backward();
            (a.gradient(), b.gradient())
        };

        let (a, b) = compute_graph();
        assert_eq!(format!("{:.4}", a), "138.8338");
        assert_eq!(format!("{:.4}", b), "645.5773");
    }

    #[test]
    fn karpathy_video_tutorial() {
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);

        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);

        let b = Value::new(6.8813735870195432);

        let x1w1 = x1.clone() * w1.clone();
        let x2w2 = x2.clone() * w2.clone();
        let x1w1x2w2 = x1w1.clone() + x2w2.clone();
        let n = x1w1x2w2.clone() + b.clone();
        let o = n.clone().tanh();

        assert_eq!(format!("{:.4}", n.data()), "0.8814");
        assert_eq!(format!("{:.4}", o.data()), "0.7071");

        o.backward();
        assert_eq!(format!("{:.4}", n.gradient()), "0.5000");
        assert_eq!(format!("{:.4}", w2.gradient()), "0.0000");
        assert_eq!(format!("{:.4}", x2.gradient()), "0.5000");
        assert_eq!(format!("{:.4}", x1.gradient()), "-1.5000");
        assert_eq!(format!("{:.4}", w1.gradient()), "1.0000");
    }

    #[test]
    fn test_nn() {
        let nn = Neuron::new(2);
        let x = [Value::new(2.0), Value::new(3.0)];

        let v = nn.call(&x[..]);
        println!("{:.4}", v.data());

        let layer = Layer::new(2, 3);
        let v = layer.call(&x[..]);

        for v in v {
            println!("{:.4}", v.data());
        }

        let mlp = MLP::new(2, &[4, 4, 1]);

        let res = mlp.call(&x[..]);
        println!("{}", res.len());
        println!("{:.4}", res[0].data());
    }
}
