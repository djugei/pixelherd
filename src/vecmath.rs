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

/// atan2 is the inverse of sin_cos()
/// and turns a (unit/normalized)-vector into rads
pub fn atan2(vec: Vector) -> f64 {
    vec[0].atan2(vec[1])
}

/// normalizes a value in radians back into the [-pi,pi] range.
/// this is required after basically any operation on (2) radians.
/// this is not simply modulo, as modulo maps pi+1 -> 1 instead of -pi+1
pub fn rad_norm(rad: f64) -> f64 {
    //todo: maybe i can express this in a nicer way, with only one modulo and no shifts
    use std::f64::consts::PI;
    // valid range:
    // [-pi, pi] -> [0, 2pi] -> % -> [-pi, pi]
    let up = ((rad + PI) % (2. * PI)) - PI;
    // [-pi, pi] -> [-2pi, 0] -> % -> [-pi, pi]
    let down = ((up - PI) % (2. * PI)) + PI;
    down
}

#[inline]
pub fn validate(v: &Vector) {
    debug_assert!(notnan(v), "NaNs found in {:?}", v);
}

pub fn notnan(v: &Vector) -> bool {
    !(v[0].is_nan() || v[1].is_nan())
}
