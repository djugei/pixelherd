//! fix-rat is a rational number with the denominator chosen at compile time.
//!
//! It has a fixed valid range.
//!
//!     # use fix_rat::Rational;
//!     type R = Rational<{ i64::MAX / 64}>;
//!
//!     let a: R = 60.into();
//!     let b: R = 5.0.into();
//!     let c = a.wrapping_add(b);
//!
//!     let c = c.to_i64();
//!     assert_eq!(c, -63);
//!
//! # Intended Use-Cases
//!
//! It is meant to replace regular floating point numbers
//!     (f64/f32)
//! in the following cases:
//!
//! 1. Requiring reliable precision,
//!     for example for handling money.
//! If you have a requirement like
//!     "x decimal/binary places after the dot"
//! then this crate might be for you.
//!
//! 2. Requiring "deterministic"
//!     (actually commutative and associative)
//! behaviour for multithreading.
//! If you want to apply updates to a value and
//! the result should be the same no matter in what order the updates were applied then
//! this crate might be for you.
//!
//! 3. Better precision if plausible value range is known.
//! Floating point numbers are multi-scalar,
//!     they can represent extremely small and extremely big numbers.
//! If your calculations stay within a known interval,
//!     for example [-1, 1],
//! fixed-rat might be able to provide better precision and performance.
//!
//! 4. todo: low precision and high performance requirements:
//! Floating point numbers only come in two variants,
//! f32 and f64, using 32 and 64 bits respectively.
//! If you do not require that much precision,
//!     if 16 or even 8 bits are enough for your usecase
//! you can store more numbers in the same space.
//!     Due to lower memory bandwidth and availability of SIMD-instructions
//! this can lead to close to a 2x or 4x respectively speed up of affected pieces of code.
//!
//! # Gotchas (tips&tricks)
//! For reliable precision:
//!     remember that you are still loosing precision on every operation,
//!     there is no way around this except bigints.
//!
//! Unlike floats Rationals have a valid range and it is easy to over/underflow it.
//! It might be advisable to choose the representable range slightly larger.
//!
//! Using rationals does not automatically make multithreaded code deterministic.
//! The determinism loss with floating point numbers happens when
//!     a calculation changes the scale (exponent) of the number.
//! Rationals are always on the same scale but you now have to deal with range overflows.
//!
//! The easiest behaviour is to use wrapping\_op.
//! It always succeeds and can be executed in any order with any values without loosing determinism.
//! This might not be sensible behaviour for your use-case though.
//!
//! The second-easiest is to use is checked\_op with unwrap.
//! This is probably fine if you have a base value that is only changed in small increments.
//! Choose a slightly bigger representable range and do the overflow handling in synchronous code,
//!     for example by clamping to the valid range
//!         (which is smaller than the representable range).
//!
//! You can not,
//!     at least not naively,
//! actually check the checked\_op,
//!     as that would generally lead to behaviour differences on different execution orders.
//! Correctly doing this is the hardest option,
//!     but might be required for correctness.
//!
//! Using saturating\_op can be perfectly valid,
//!     but you need to be careful that the value can only be pushed into one direction
//!         (either towards max or towards min).
//! Otherwise different execution orders lead to different results.
//! Reminder that adding negative values is also subtraction!
//!
//! Assuming [-10,10]:
//!     `9 + 2 = 10, 10 - 1 =  9`
//! but
//!     `9 - 1 =  8,  8 + 2 = 10`
//! 9 != 10.
//!
//! Moving through different scales,
//!     mainly by multiplying/dividing
//! costs more precision than you might be used from floating point numbers.
//! For example diving by 2 costs no precision in floating point numbers,
//!     it simply decreases the exponent by one.
//! In rationals it costs one bit of precsion.
//! Remember that rationals start out with 63 bits though,
//! while f64 only has 53.
//!
//! # Implementation
//! This is a super simple wrapper around an integer,
//! basically all operations are passed straight through.
//! So an addition of two rationals is really just a simple integer addition.
//!
//! Converting an integer/float to a rational simply multiplies it by the chosen DENOM,
//! rational to integer/float divides.
//!
//! The code is very simple.
//! The main value of this crate is the tips&tricks and ease of use.
//!
//! # todos/notes
//! Currently being generic over intergers is a bit.. annoying. Being generic over intergers while
//! also taking a value of that type as a const generic is.. currently not typechecking. so
//! supporting usecase 4 would need some macroing (i&u 8,16,32,64). For now its just always i64.
//!
//! I should probably provide some atomic operations for comfort, at least add&sub.
//! Even though they are simply identical to just adding/subbing on the converted atomic.
//!
//! Currently there is no rat-rat interaction with different denominatiors.
//! That could be improved,
//! but might need to wait for better const generics.
//!
//! # nightly
//! This crate very much inherently relies on const generics (min\_const\_generics).

#![no_std]

pub use nightly::Rational;

/// Can store -10 to 10 with a bit of wiggle room.
pub type TenRat = Rational<{ i64::MAX / 16 }>;

/// Can store -100 to 100 with a bit of wiggle room.
pub type HundRat = Rational<{ i64::MAX / 128 }>;

mod nightly {
    #[cfg(feature = "serde1")]
    use serde as sd;
    /// A ratonal number with a fixed denominator,
    /// therefore `size_of<Rational>() == size_of<i64>()`.
    ///
    /// Plus operations have more intuitive valid ranges
    ///
    /// If you want to represent numbers in range `-x` to `x`
    /// choose DENOM as `i64::MAX / x`.
    ///
    /// [Rational] then subdivides the whole range into i64::MAX equally-sized parts.
    /// Smaller operations are lost,
    /// going outside the range overflows.
    ///
    /// DENOM needs to be positive or you will enter the bizarro universe.
    ///
    /// I would strongly recommend to choose DENOM as `1 << x` for x in 0..63.
    ///
    /// The regular operations (+,-, *, /) behave just like on a regular integer,
    ///     panic on overflow in debug mode,
    ///     wrapping in release mode.
    /// Use the wrapping_op, checked_op or saturating_op methods
    ///     to explicitly chose a behaviour.
    #[derive(Default, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    #[cfg_attr(feature = "serde1", derive(sd::Serialize, sd::Deserialize))]
    pub struct Rational<const DENOM: i64> {
        numer: i64,
    }
    impl<const DENOM: i64> Rational<DENOM> {
        /// Returns the underlying integer type,
        /// for example for storing in an atomic number.
        ///
        /// This should compile to a no-op.
        pub fn to_storage(self) -> i64 {
            self.numer
        }
        /// Builds from the underlying integer type,
        /// for example after retrieving from an atomic number.
        ///
        /// This should compile to a no-op.
        ///
        /// Use [from_int](Self::from_int) if you have an integer that you want to convert to a rational.
        pub fn from_storage(storage: i64) -> Self {
            Self { numer: storage }
        }

        /// Converts an integer to a Rational.
        pub fn from_int(i: i64) -> Self {
            Self { numer: i * DENOM }
        }
        /// Since rational numbers can not represent inf, nan and other fuckery,
        /// this returns None if the input is wrong.
        ///
        /// This will loose precision,
        /// try to only convert at the start and end of your calculation.
        pub fn aprox_float_fast(f: f64) -> Option<Self> {
            use core::num::FpCategory as Cat;
            if let Cat::Subnormal | Cat::Normal | Cat::Zero = f.classify() {
            } else {
                return None;
            }

            // this is really not very accurate
            // as the expansion step kills a lot of accuracy the float might have had
            // (denom is really big, int_max/representable_range)
            let expanded = f * (DENOM as f64);
            Some(Self {
                numer: expanded as i64,
            })
        }

        /// Assumes denom to be a power of two.
        /// kind of experimental.
        #[doc(hidden)]
        pub fn aprox_float(f: f64) -> Option<Self> {
            use core::num::FpCategory as Cat;
            match f.classify() {
                //fixme: im reasonably sure that subnormal needs to be handled different
                //(implicit 1 or something along those lines)
                Cat::Subnormal | Cat::Normal => {}
                Cat::Zero => return Self::from(0).into(),
                _ => return None,
            }
            use num_traits::float::FloatCore;
            let (mant, f_exp, sign) = f.integer_decode();
            //exp is << or >> on the mant depending on sign
            let d_exp = 64 - (DENOM as u64).leading_zeros();
            //let rest = DENOM - (1 << d_exp);

            let exp = f_exp + (d_exp as i16);
            let neg = exp.is_negative();
            let exp = exp.abs() as u32;
            let numer = if !neg {
                // make sure we have enough headroom
                // cheked_shl/r does not! do this check
                if mant.leading_zeros() < exp {
                    return None;
                }
                mant << exp
            } else {
                // not checking for "bottom"-room here as we are
                // "just" loosing precision, not orders of magnitude
                mant >> exp
            };
            let numer = numer as i64;
            let numer = if sign.is_negative() { -numer } else { numer };
            // fixme: do something about the rest??
            // hm, or just allow 1<<x as denom, sounds senible too
            Self { numer }.into()
        }

        /// This will loose precision,
        /// try to only convert at the start and end of your calculation.
        pub fn to_f64(self) -> f64 {
            //todo: can i do a *thing* with this to get some more precision?
            //(i.e. bitbang the float?)
            self.numer as f64 / DENOM as f64
        }

        /// this will integer-round, potentially loosing a lot of precision.
        pub fn to_i64(self) -> i64 {
            self.numer / DENOM
        }

        pub fn clamp(self, low: Self, high: Self) -> Self {
            self.max(low).min(high)
        }
        /// The maximum representable number.
        ///
        /// Note that unlike floats rationals do not have pos/neg inf.
        pub const fn max() -> Self {
            Self { numer: i64::MAX }
        }
        /// The minimum representable number.
        ///
        /// Note that unlike floats rationals do not have pos/neg inf.
        pub const fn min() -> Self {
            Self { numer: i64::MIN }
        }

        pub fn checked_add(self, other: Self) -> Option<Self> {
            Self {
                numer: self.numer.checked_add(other.numer)?,
            }
            .into()
        }
        pub fn checked_mul(self, other: Self) -> Option<Self> {
            Self {
                numer: self.numer.checked_mul(other.numer)?,
            }
            .into()
        }
        pub fn checked_sub(self, other: Self) -> Option<Self> {
            Self {
                numer: self.numer.checked_sub(other.numer)?,
            }
            .into()
        }
        pub fn checked_div(self, other: Self) -> Option<Self> {
            Self {
                numer: self.numer.checked_div(other.numer)?,
            }
            .into()
        }

        pub fn wrapping_add(self, other: Self) -> Self {
            Self {
                numer: self.numer.wrapping_add(other.numer),
            }
        }
        pub fn wrapping_mul(self, other: Self) -> Self {
            Self {
                numer: self.numer.wrapping_mul(other.numer),
            }
        }
        pub fn wrapping_sub(self, other: Self) -> Self {
            Self {
                numer: self.numer.wrapping_sub(other.numer),
            }
        }
        pub fn wrapping_div(self, other: Self) -> Self {
            Self {
                numer: self.numer.wrapping_div(other.numer),
            }
        }

        /// Don't use this in parallel code if other parallel code is also subtracting,
        /// otherwise you loose determinism.
        ///
        /// ```
        /// use fix_rat::Rational;
        /// let max = Rational::<{1024}>::max();
        /// let one = Rational::<{1024}>::from_int(1);
        /// assert_ne!(max.saturating_add(one)-max, (max-max).saturating_add(one));
        /// ```
        pub fn saturating_add(self, other: Self) -> Self {
            Self {
                numer: self.numer.saturating_add(other.numer),
            }
        }
        pub fn saturating_mul(self, other: Self) -> Self {
            Self {
                numer: self.numer.saturating_mul(other.numer),
            }
        }
        pub fn saturating_sub(self, other: Self) -> Self {
            Self {
                numer: self.numer.saturating_sub(other.numer),
            }
        }
    }

    impl<const DENOM: i64> From<f64> for Rational<DENOM> {
        fn from(o: f64) -> Self {
            // apparently _fast is not less precise than "regular" so using that for now
            // might change at a moments notice though
            Self::aprox_float_fast(o).unwrap()
        }
    }
    impl<const DENOM: i64> From<i64> for Rational<DENOM> {
        fn from(o: i64) -> Self {
            Self::from_int(o)
        }
    }
    impl<const DENOM: i64> core::ops::Add for Rational<DENOM> {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            Self {
                numer: self.numer + other.numer,
            }
        }
    }
    impl<const DENOM: i64> core::ops::Sub for Rational<DENOM> {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            Self {
                numer: self.numer - other.numer,
            }
        }
    }
    impl<const DENOM: i64> core::ops::Mul<i64> for Rational<DENOM> {
        type Output = Self;
        fn mul(self, other: i64) -> Self {
            Self {
                numer: self.numer * other,
            }
        }
    }
    impl<const DENOM: i64> core::ops::Div<i64> for Rational<DENOM> {
        type Output = Self;
        fn div(self, other: i64) -> Self {
            Self {
                numer: self.numer / other,
            }
        }
    }

    impl<const DENOM: i64> core::iter::Sum for Rational<DENOM> {
        fn sum<I>(i: I) -> Self
        where
            I: Iterator<Item = Self>,
        {
            i.fold(Self::from(0), |sum, new| sum + new)
        }
    }

    // well guess no mul/div then
    /*
    impl<const DENOMS: i64, const DENOMO: i64> core::ops::Mul<Rational<DENOMO>> for Rational<DENOMS> {
        type Output = Rational<{ DENOMS * DENOMO }>;
        fn mul(self, other: Self) -> Self {
            Self::Output {
                numer: self.numer * other.numer,
            }
        }
    }
    */
}
#[test]
fn converts() {
    let _tenrat: TenRat = 0.0.into();
    let _tenrat: TenRat = 1.0.into();
    let _tenrat: TenRat = (-1.0).into();
    let _tenrat: TenRat = 10.0.into();
    let _tenrat: TenRat = (-10.0).into();
}

#[test]
fn precision() {
    type R = Rational<{ i64::MAX / (1 << 10) }>;
    extern crate std;
    use std::dbg;
    let f = 640.132143234189097_f64;
    let r: R = nightly::Rational::aprox_float(f).unwrap();
    let r2: R = nightly::Rational::aprox_float_fast(f).unwrap();
    let rf = r.to_f64();
    // ok so turns out that the fast conversion is actually not worse.
    // i guess multiplying/diving by 2potn is kinda what floats are good at
    let rl = r2.to_f64();

    let absdiff = (f - rf).abs();
    dbg!(f, r, r2, rf, rl, absdiff);
    assert!(absdiff < 1e20);
}

#[test]
fn displaytest() {
    let tenrat: TenRat = (-10.0).into();
    extern crate std;
    use std::println;
    println!("{:#?}", tenrat);
}

#[cfg(feature = "serde1")]
#[test]
fn serde_test() {
    let r: TenRat = 3.0.into();
    use bincode;
    let s = bincode::serialize(&r).unwrap();
    let d = bincode::deserialize(&s).unwrap();
    assert_eq!(r, d);
}
