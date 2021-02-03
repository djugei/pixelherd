# Fix(ed) Rat(ional)

<!-- cargo-sync-readme start -->

fix-rat is a rational number with the denominator chosen at compile time.

It has a fixed valid range.

    # use fix_rat::Rational;
    type R = Rational<{ i64::MAX / 64}>;

    let a: R = 60.into();
    let b: R = 5.0.into();
    let c = a.wrapping_add(b);

    let c = c.to_i64();
    assert_eq!(c, -63);

# Intended Use-Cases

It is meant to replace regular floating point numbers
    (f64/f32)
in the following cases:

1. Requiring reliable precision,
    for example for handling money.
If you have a requirement like
    "x decimal/binary places after the dot"
then this crate might be for you.

2. Requiring "deterministic"
    (actually commutative and associative)
behaviour for multithreading.
If you want to apply updates to a value and
the result should be the same no matter in what order the updates were applied then
this crate might be for you.

3. Better precision if plausible value range is known.
Floating point numbers are multi-scalar,
    they can represent extremely small and extremely big numbers.
If your calculations stay within a known interval,
    for example [-1, 1],
fixed-rat might be able to provide better precision and performance.

4. todo: low precision and high performance requirements:
Floating point numbers only come in two variants,
f32 and f64, using 32 and 64 bits respectively.
If you do not require that much precision,
    if 16 or even 8 bits are enough for your usecase
you can store more numbers in the same space.
    Due to lower memory bandwidth and availability of SIMD-instructions
this can lead to close to a 2x or 4x respectively speed up of affected pieces of code.

# Gotchas (tips&tricks)
For reliable precision:
    remember that you are still loosing precision on every operation,
    there is no way around this except bigints.

Unlike floats Rationals have a valid range and it is easy to over/underflow it.
It might be advisable to choose the representable range slightly larger.

Using rationals does not automatically make multithreaded code deterministic.
The determinism loss with floating point numbers happens when
    a calculation changes the scale (exponent) of the number.
Rationals are always on the same scale but you now have to deal with range overflows.

The easiest behaviour is to use wrapping\_op.
It always succeeds and can be executed in any order with any values without loosing determinism.
This might not be sensible behaviour for your use-case though.

The second-easiest is to use is checked\_op with unwrap.
This is probably fine if you have a base value that is only changed in small increments.
Choose a slightly bigger representable range and do the overflow handling in synchronous code,
    for example by clamping to the valid range
        (which is smaller than the representable range).

You can not,
    at least not naively,
actually check the checked\_op,
    as that would generally lead to behaviour differences on different execution orders.
Correctly doing this is the hardest option,
    but might be required for correctness.

Using saturating\_op can be perfectly valid,
    but you need to be careful that the value can only be pushed into one direction
        (either towards max or towards min).
Otherwise different execution orders lead to different results.
Reminder that adding negative values is also subtraction!

Assuming [-10,10]:
    `9 + 2 = 10, 10 - 1 =  9`
but
    `9 - 1 =  8,  8 + 2 = 10`
9 != 10.

Moving through different scales,
    mainly by multiplying/dividing
costs more precision than you might be used from floating point numbers.
For example diving by 2 costs no precision in floating point numbers,
    it simply decreases the exponent by one.
In rationals it costs one bit of precsion.
Remember that rationals start out with 63 bits though,
while f64 only has 53.

# Implementation
This is a super simple wrapper around an integer,
basically all operations are passed straight through.
So an addition of two rationals is really just a simple integer addition.

Converting an integer/float to a rational simply multiplies it by the chosen DENOM,
rational to integer/float divides.

The code is very simple.
The main value of this crate is the tips&tricks and ease of use.

# todos/notes
Currently being generic over intergers is a bit.. annoying. Being generic over intergers while
also taking a value of that type as a const generic is.. currently not typechecking. so
supporting usecase 4 would need some macroing (i&u 8,16,32,64). For now its just always i64.

I should probably provide some atomic operations for comfort, at least add&sub.
Even though they are simply identical to just adding/subbing on the converted atomic.

Currently there is no rat-rat interaction with different denominatiors.
That could be improved,
but might need to wait for better const generics.

# nightly
This crate very much inherently relies on const generics (min\_const\_generics).

<!-- cargo-sync-readme end -->

# Contributing
See README.md at the root of the repository.
