/// contains some very simple helpers for vector rotation mainly

pub type Vector = [f64; 2];

/// calculates the length of a vector
pub fn len(inp: Vector) -> f64 {
    validate(&inp);
    len2(inp).sqrt()
}

/// calculates length*length of a vector
/// which conveniently avoids a sqrt
pub fn len2(inp: Vector) -> f64 {
    validate(&inp);
    (inp[0] * inp[0]) + (inp[1] * inp[1])
}

/// turns the input into a vector that has length 1
pub fn norm(mut inp: Vector) -> Vector {
    validate(&inp);
    let len = len(inp);
    inp[0] /= len;
    inp[1] /= len;
    inp
}

/// rotates two vectors aroud each other.
/// the argument order does not matter, but by convention
/// base is the vector you want to rotate, while rot describes the rotation
///
/// if |rot| == 1 then |base| == |result|
/// if len(rot) == 1 then len(base) == len(result)
///
/// discounting floating point errors
/// therefore if you want to preserve the length you should norm the rot first.
pub fn rotate(base: Vector, rot: Vector) -> Vector {
    validate(&base);
    validate(&rot);
    let real = (base[0] * rot[0]) - (base[1] * rot[1]);
    let imag = (base[0] * rot[1]) + (base[1] * rot[0]);
    [real, imag]
}

/// componet-wise addition
pub fn add(mut a: Vector, b: Vector) -> Vector {
    validate(&a);
    validate(&b);
    a[0] += b[0];
    a[1] += b[1];
    a
}

/// componet-wise sub
pub fn sub(mut a: Vector, b: Vector) -> Vector {
    validate(&a);
    validate(&b);
    a[0] -= b[0];
    a[1] -= b[1];
    a
}

/// scales a vector by a scalar
pub fn scale(mut a: Vector, scalar: f64) -> Vector {
    validate(&a);
    debug_assert!(!scalar.is_nan());
    a[0] *= scalar;
    a[1] *= scalar;
    a
}

/// relative angle, in radians, range -pi - pi
#[allow(unused)]
pub fn angle_between(base: Vector, other: Vector) -> f64 {
    let delta = sub(other, base);
    let n = norm(delta);
    angle(n)
}

/// angle on the unit cirle, input needs to be normalize
/// otherwise you get NaNs
/// to be more precize actually only the x component has to be normalized
pub fn angle(norm: Vector) -> f64 {
    let x = norm[0];
    let rad = x.asin();
    let ypos = norm[1].is_sign_positive();

    // fixme: this feels insanely ugly
    if ypos {
        rad
    } else {
        std::f64::consts::PI.copysign(x) - rad
    }
}

#[test]
fn angle_test() {
    let steps = 100_000;
    let inc = 1. / (steps as f64);
    let pi = std::f64::consts::PI;
    for i in -steps..steps {
        let num = (i as f64) * inc;
        let pinum = num * pi;
        let (x, y) = pinum.sin_cos();

        let a = angle([x, y]);

        let d = (pinum - a).abs();
        assert!(
            d < 0.00001,
            "pinum: {}, a: {}, x: {}, d: {}",
            pinum,
            a,
            x,
            d
        );
    }
}

#[inline]
pub fn validate(v: &Vector) {
    debug_assert!(notnan(v), "NaNs found in {:?}", v);
}

pub fn notnan(v: &Vector) -> bool {
    !(v[0].is_nan() || v[1].is_nan())
}
