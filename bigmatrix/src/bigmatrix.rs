// this is straight up ripped from serde-big-array
// im not sure if this is compex enough to warrant copyright, but if so its MIT
use core::fmt;
use core::marker::PhantomData;
use core::result;
use serde::de::{Deserialize, Deserializer, Error, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeTuple, Serializer};

pub trait BigMatrix<'de>: Sized {
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where
        S: Serializer;
    fn deserialize<D>(deserializer: D) -> result::Result<Self, D::Error>
    where
        D: Deserializer<'de>;
}
impl<'de, T, const N: usize, const M: usize> BigMatrix<'de> for [[T; M]; N]
where
    T: Default + Copy + Serialize + Deserialize<'de>,
{
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // serializing into a flattened matrix
        let mut seq = serializer.serialize_tuple(N * M)?;
        for elem in &self[..] {
            for inner_elem in elem {
                seq.serialize_element(inner_elem)?;
            }
        }
        seq.end()
    }

    fn deserialize<D>(deserializer: D) -> result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MatrixVisitor<T> {
            element: PhantomData<T>,
        }

        impl<'de, T, const N: usize, const M: usize> Visitor<'de> for MatrixVisitor<[[T; M]; N]>
        where
            T: Default + Copy + Deserialize<'de>,
        {
            type Value = [[T; M]; N];

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                macro_rules! write_len {
                    ($l:literal) => {
                        write!(formatter, concat!("an array of length ", $l))
                    };
                    ($l:tt) => {
                        write!(formatter, "an array of length {}", $l)
                    };
                }

                write_len!(N)
            }

            fn visit_seq<A>(self, mut seq: A) -> result::Result<[[T; M]; N], A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut arr: [[T; M]; N] = [[T::default(); M]; N];
                for i in 0..N {
                    for j in 0..M {
                        arr[i][j] = seq
                            .next_element()?
                            .ok_or_else(|| Error::invalid_length(i, &self))?;
                    }
                }
                Ok(arr)
            }
        }

        let visitor = MatrixVisitor {
            element: PhantomData,
        };
        // The allow is needed to support (32 + 33) like expressions
        #[allow(unused_parens)]
        deserializer.deserialize_tuple(N * M, visitor)
    }
}
