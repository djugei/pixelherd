#![no_std]
#![feature(const_generics)]
#![cfg(feature = "nightly")]
#![allow(incomplete_features)]
#![cfg(feature = "nightly")]

#[cfg(feature = "nightly")]
pub use nightly::Rational;

#[cfg(feature = "nightly")]
///can store -10 to 10 with a bit of wiggle room
pub type TenRat = Rational<{ i64::MAX / 16 }>;

#[cfg(feature = "nightly")]
///can store -100 to 100 with a bit of wiggle room
pub type HundRat = Rational<{ i64::MAX / 128 }>;

mod nightly {
    #[cfg(feature = "serde1")]
    use serde as sd;
    /// a ratonal number with a fixed denom, therefore
    /// sizeof\<Rational> == sizeof\<i64>. plus operations have more intuitive valid ranges
    ///
    ///
    /// if you want to represent numbers in range -2.exp(x) to 2.exp(x)
    /// choose denom as {i64::MAX / (1 << x)}
    ///
    /// Rational then subdivides the whole range into i64::MAX equally-sized parts
    /// smaller operations are lost, going outside the range overflows.
    ///
    /// denom needs to be positive or you will enter the bizarro universe
    /// i would strongly recommend to choose denom as a 2.exp(x)
    #[derive(Default, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    #[cfg_attr(feature = "serde1", derive(sd::Serialize, sd::Deserialize))]
    pub struct Rational<const DENOM: i64> {
        numer: i64,
    }
    impl<const DENOM: i64> Rational<DENOM> {
        pub fn from_int(i: i64) -> Self {
            Self { numer: i * DENOM }
        }
        /// since rationals can not represent inf, nan and other fuckery this returns none if the
        /// input is wrong
        /// really not very precise
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

        /// kinda assumes that denom is 2.exp(x) otherwise (more) precision is lost
        /// otherwise the conversion is actually not very lossy at all, except for very small
        /// values (that are not representable in that precision)
        ///
        /// try to input values that are actually inside the valid range, otherwise you will get
        /// None back
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
        /// rounding errors
        pub fn to_f64(self) -> f64 {
            //todo: can i do a *thing* with this to get some more precision?
            //(i.e. bitbang the float?)
            self.numer as f64 / DENOM as f64
        }
        pub fn clamp(self, low: Self, high: Self) -> Self {
            self.max(low).min(high)
        }
        pub const fn max() -> Self {
            Self { numer: i64::MAX }
        }
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
