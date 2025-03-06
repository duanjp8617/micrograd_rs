use micrograd::*;

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

    let mlp = MLP::new(3, &[4, 4, 1], false);
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
            w.set_data(w.data() + (-0.01) * w.gradient());
        }

        println!("iteration {}: loss {:.4}", i, loss.data());
    }

    for y_pred in y_preds {
        println!("{:.4}", y_pred.data());
    }
}
