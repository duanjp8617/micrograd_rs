use std::cell::RefCell;
use std::collections::HashSet;
use std::convert::From;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::rc::Rc;

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
}
