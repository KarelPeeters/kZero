#[derive(Debug, Copy, Clone, PartialEq)]
struct Welford {
    count: usize,
    mean: f32,
    m2: f32,
}

impl Welford {
    fn empty() -> Self {
        Welford {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    fn single(value: f32) -> Self {
        Welford {
            count: 1,
            mean: value,
            m2: 0.0,
        }
    }

    fn series(values: &[f32]) -> Self {
        if values.len() == 0 {
            return Self::empty();
        }

        let count = values.len();
        let mean = values.iter().copied().sum::<f32>() / count as f32;
        let m2 = values.iter().map(|&f| (f - mean).powi(2)).sum();
        Welford { count, mean, m2 }
    }

    fn append(&mut self, value: f32) {
        println!("Appending {} to {:?}", value, self);

        let count = self.count + 1;
        let delta = value - self.mean;
        let mean = self.mean + delta / count as f32;
        let m2 = self.m2 + delta * (value - mean);

        *self = Welford { count, mean, m2 };
        println!("  result {:?}", self);
    }

    fn combine(&self, other: Welford) -> Self {
        let a = *self;
        let b = other;

        println!("Combining {:?} and {:?}", a, b);

        let count = a.count + b.count;
        let delta = b.mean - a.mean;

        let div_count = if count == 0 { 1 } else { count };
        let mean = a.mean + delta * (b.count as f32 / div_count as f32);
        let m2 = a.m2 + b.m2 + delta * delta * ((a.count * b.count) as f32 / div_count as f32);

        let result = Welford { count, mean, m2 };

        println!("  result {:?}", result);

        result
    }

    fn finish(self) -> (f32, f32) {
        if self.count == 0 {
            (f32::NAN, f32::NAN)
        } else {
            (self.mean, self.m2 / self.count as f32)
        }
    }
}

#[test]
fn append() {
    let mut result = Welford::single(1.0);
    result.append(2.0);

    let expected = Welford::series(&[1.0, 2.0]);
    assert_eq!(result, expected);
}

#[test]
fn combine() {
    let a = Welford::single(1.0);
    let b = Welford::single(2.0);

    let result = a.combine(b);
    let expected = Welford::series(&[1.0, 2.0]);

    assert_eq!(result, expected);
}

#[test]
fn combine_empty() {
    let a = Welford::series(&[1.0, 2.0]);
    let b = Welford::empty();

    assert_eq!(a, a.combine(b));
    assert_eq!(a, b.combine(a));
}

#[test]
fn combine_vs_append() {
    let a = Welford::series(&[1.0, 2.0, 3.0]);
    let b = Welford::single(4.0);
    let expected = Welford::series(&[1.0, 2.0, 3.0, 4.0]);

    let combined = a.combine(b);

    let mut appended = a;
    appended.append(4.0);

    assert_eq!(expected, combined);
    assert_eq!(expected, appended);
}

#[test]
fn append_multi() {
    let mut curr = Welford::empty();
    curr.append(1.0);
    curr.append(2.0);
    curr.append(3.0);
    curr.append(4.0);

    let expected = Welford::series(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(curr, expected);
}

#[test]
fn nan_carried() {
    let mut curr = Welford::empty();
    curr.append(1.0);
    curr.append(f32::NAN);
    curr.append(3.0);
    curr.append(4.0);

    assert!(curr.mean.is_nan());
    assert!(curr.m2.is_nan());
    assert!(curr.finish().0.is_nan());
    assert!(curr.finish().1.is_nan());
}

#[test]
fn empty_empty() {
    let a = Welford::empty();
    let b = Welford::empty();

    assert_eq!(a, a.combine(b));

    assert!(a.finish().0.is_nan());
    assert!(a.finish().1.is_nan());
}
