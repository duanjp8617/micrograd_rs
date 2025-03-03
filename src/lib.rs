use std::cell::RefCell;
use std::convert::From;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::rc::Rc;

type GradFunc = Box<dyn Fn(f64)>;
type ObjPtr = Rc<RefCell<ValueInner>>;

struct ValueInner {
    data: f64,
    gradient: f64,
    prevs: Vec<(ObjPtr, GradFunc)>,
}

impl ValueInner {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            gradient: 0.0,
            prevs: vec![],
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

    pub fn powf(self, exponent: f64) -> Value {
        let x = self.0.clone();
        // d(x^exponent)/dx -> exponent * (x)^(exponent-1) * parent_grad
        let x_grad_func = Box::new(move |parent_grad: f64| {
            x.borrow_mut().gradient +=
                exponent * (x.borrow().data.powf(exponent - 1.0)) * parent_grad
        });

        let val = ValueInner {
            data: self.data().powf(exponent),
            gradient: 0.0,
            prevs: vec![(self.0.clone(), x_grad_func)],
        };

        Value(Rc::new(RefCell::new(val)))
    }

    pub fn powi(self, exponent: i32) -> Value {
        self.powf(exponent.into())
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
        // d(x + y)/dx -> 1 * parent_grad
        let x_grad_func = Box::new(move |parent_grad: f64| x.borrow_mut().gradient += parent_grad);

        let y = rhs.0.clone();
        // d(x + y)/dy -> 1 * parent_grad
        let y_grad_func = Box::new(move |parent_grad: f64| y.borrow_mut().gradient += parent_grad);

        let val = ValueInner {
            data: self.data() + rhs.data(),
            gradient: 0.0,
            prevs: vec![(self.0.clone(), x_grad_func), (rhs.0.clone(), y_grad_func)],
        };

        Value(Rc::new(RefCell::new(val)))
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        let x = self.0.clone();
        // d(-x)/dx -> -1 * parent_grad
        let x_grad_func = Box::new(move |parent_grad: f64| x.borrow_mut().gradient -= parent_grad);

        let val = ValueInner {
            data: -self.data(),
            gradient: 0.0,
            prevs: vec![(self.0.clone(), x_grad_func)],
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
        let x_grad_func = Box::new(move |parent_grad: f64| {
            x.borrow_mut().gradient += y.borrow().data * parent_grad
        });

        let x = self.0.clone();
        let y = rhs.0.clone();
        // d(x * y)/dy -> x * parent_grad
        let y_grad_func = Box::new(move |parent_grad: f64| {
            y.borrow_mut().gradient += x.borrow().data * parent_grad
        });

        let val = ValueInner {
            data: self.data() * rhs.data(),
            gradient: 0.0,
            prevs: vec![(self.0.clone(), x_grad_func), (rhs.0.clone(), y_grad_func)],
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
