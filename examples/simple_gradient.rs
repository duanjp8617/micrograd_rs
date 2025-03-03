use micrograd::*;

fn main() {
    let a = Value::new(-4.0);
    let b = Value::new(2.0);
    let mut c = a.clone() + b.clone();
    let d = a.clone() * b.clone() + b.clone().powi(3);
    println!("{}", c.data());
    c += (c.clone() + 1.into());

    println!("{}", c.data());
}
