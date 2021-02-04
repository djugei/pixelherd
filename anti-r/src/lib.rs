//! Anti-R contains a alternative spatial data structure that outperforms R-Trees in many cases.
//!
//! # Performance:
//! R-Trees and anti-r have the same O(n) complexity for all operations,
//! log(n) for searching and updating, n\*log(n) for creation.
//!
//! They only differ by constant factors,
//! either x or y in O(log\_b(n+x)+y)
//! and the base of the logarithm,
//! which is 2 for Anti-R and configurable for R-Tree, generally 3-6.
//!
//! Anti-R is always faster at updating all elements and bulk-loading by a constant factor,
//! therefore it is more noticeable for small n.
//!
//! Full updates and bulk-loads are equivalent in speed for Anti-R.
//! For R-Trees full updates are never worth it,
//! a full reconstruction is simply faster.
//!
//! Zero to a bit more than 100\_000 elements are faster to query for Anti-R.
//! R-Trees start winning at above 200\_000 elements.
//! This is probably when (on the benchmarking machine) L1-cache is overrun.
//!
//! R-Trees might be catching up quicker if the elements are weirdly distributed.
//!
//! See the bench directory and the output of cargo bench (target/criterion) for more details.
//!
//! Notice that this has been benched against the rstar crate,
//! which might not be the fastest implementation of an R-Tree in existence.
//! The benchmark results are exactly as expected though.

#![no_std]
#[cfg(test)]
mod tests;

/// A slice that supports spatial queries.
///
/// The slice is basically just sorted by K.
/// If K is a position in space this means its sorted by its coordinates
/// which allows for spatial queries with logarithmic complexity.
///
/// If you want to modify elements just let this go out of scope, modify the base slice and rebuild
/// the SpatSlice, potentially using new_unchecked if you uphold ordering.
/// If you opt into the alloc feature you can also use [SpatVec].
///
/// The trait bound on the K for most of the functions is PartialOrd.
/// This is a lie.
/// If a comparison fails the function will panic.
/// Therefore the real trait bound would be Ord.
/// Its just insanely inconvenient to have to deal with that for floats
/// which are probably the mainly used data type for this.
#[derive(PartialEq, Debug)]
pub struct SpatSlice<'a, K, V> {
    inner: &'a [(K, V)],
}

impl<'a, K, V> SpatSlice<'a, K, V>
where
    K: PartialOrd,
{
    /// Creates a new SpatSlice, by first putting base into sorted order.
    pub fn new(base: &'a mut [(K, V)]) -> Self {
        base.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self::new_unchecked(base)
    }
    /// When calling this instead of new you guarantee that the base is already sorted by K.
    /// An unsorted base will result in logic errors, not unsafety
    pub fn new_unchecked(base: &'a [(K, V)]) -> Self {
        Self { inner: base }
    }

    /// Searches for all entries that are enclosed between the edges min and max.
    /// min must be smaller than max on all coordinates for this to work.
    /// If you want to query for entities enclosed by arbitrary points
    /// do an element-wise swap:
    ///     
    ///     let mut min = [-1, -2,  3];
    ///     let mut max = [ 1,  2, -3];
    ///     for (min, max) in min.iter_mut().zip(&mut max) {
    ///         if min > max {
    ///             core::mem::swap(min, max);
    ///         }
    ///     }
    ///     assert!(min.iter().zip(&max).all(|(min,max)| min<=max));
    ///  
    pub fn query_aabb(&self, min: &K, max: &K) -> (usize, usize) {
        // todo: in theory there is no need to do two queries, a slightly addapted binary search
        // is absolutely able to search for ranges
        let start = self
            .inner
            .binary_search_by(|probe| probe.0.partial_cmp(min).unwrap());
        let start = either(start);
        // only search remainder, min is < max
        let data = &self.inner[start..];
        let end = data.binary_search_by(|probe| probe.0.partial_cmp(max).unwrap());
        let end = either(end);
        (start, start + end)
    }

    /// This is literally just a binary search,
    /// as in slice.binary_search().
    pub fn search_point(&self, p: &K) -> Result<usize, usize> {
        self.inner
            .binary_search_by(|probe| probe.0.partial_cmp(p).unwrap())
    }
}

impl<'a, K, V> core::ops::Deref for SpatSlice<'a, K, V> {
    type Target = &'a [(K, V)];
    fn deref(&self) -> &&'a [(K, V)] {
        &self.inner
    }
}

impl<'a, V> SpatSlice<'a, [f64; 2], V> {
    /// Searches for all entries within a specified range around a center point.
    ///
    /// Returns an Iterator containing (distance, &(K, V)).
    /// Excludes the point itself from the query if an exact match exists
    /// (distance will never be 0).
    ///
    /// Dist2 is the squared distance,
    /// so if you input 100 it will search for things in range 10.
    /// If you want to search for range 100 you have to input 100*100
    ///
    /// This is only implemented for 2d vectors, as const generics are currently unstable,
    /// and I did not want to create a Distance and scalarsub/add trait.
    /// This function is rather simple though so you can easily implement it yourself
    /// if you need a different K.
    pub fn query_distance<'c>(
        &'a self,
        center: &'c [f64; 2],
        dist2: f64,
    ) -> impl Iterator<Item = (f64, &'a ([f64; 2], V))> + 'c
    where
        'a: 'c,
    {
        let min = [center[0] - dist2, center[1] - dist2];
        let max = [center[0] + dist2, center[1] + dist2];
        let (start, end) = self.query_aabb(&min, &max);
        let matches = self.inner[start..end].iter();
        matches
            .map(move |kv| {
                let x = center[0] - kv.0[0];
                let y = center[1] - kv.0[1];
                let len2 = (x * x) + (y * y);
                (len2, kv)
            })
            .filter(move |(d, _)| *d < dist2)
            .filter(|(d, _)| *d != 0.)
    }
}

pub fn either<T>(i: Result<T, T>) -> T {
    match i {
        Ok(i) | Err(i) => i,
    }
}

#[cfg(feature = "alloc")]
pub use vec::SpatVec;

#[cfg(feature = "alloc")]
mod vec {
    extern crate alloc;
    use crate::SpatSlice;
    use alloc::vec::Vec;

    /// A vector whose elements are stored in sorted order at all times.
    ///
    /// Adds modification capabilities to [SpatSlice].
    #[derive(PartialEq, Debug, Clone, Eq, Ord, PartialOrd)]
    pub struct SpatVec<K, V> {
        inner: Vec<(K, V)>,
    }

    impl<K, V> SpatVec<K, V>
    where
        K: PartialOrd,
    {
        /// Turns a Vec into a SpatVec by sorting its elements.
        pub fn new_from(mut v: Vec<(K, V)>) -> Self {
            SpatSlice::new(&mut v);
            Self::new_from_unchecked(v)
        }
        /// Turns a Vec into a SpatVec.
        /// Caller guarantees elements to be sorted by K.
        pub fn new_from_unchecked(v: Vec<(K, V)>) -> Self {
            Self { inner: v }
        }
        pub fn into_inner(self) -> Vec<(K, V)> {
            self.inner
        }
        /// Applies f and then re-sorts
        pub fn map<F>(&mut self, f: F)
        where
            F: FnMut(&mut (K, V)),
        {
            self.inner.iter_mut().for_each(f);
            SpatSlice::new(&mut self.inner);
        }

        /// Keeps all elements for which f returns true.
        pub fn retain<F>(&mut self, f: F)
        where
            F: FnMut(&(K, V)) -> bool,
        {
            // deleting stuff does not change its order
            self.inner.retain(f)
        }
        pub fn as_spat_slice(&self) -> SpatSlice<'_, K, V> {
            self.into()
        }

        /// Insert a new element, retaining sort order.
        ///
        /// If one or more elements with the same key already exist, their order is unspecified.
        pub fn insert(&mut self, kv: (K, V)) {
            let p = crate::either(self.as_spat_slice().search_point(&kv.0));
            self.inner.insert(p, kv);
        }

        /// Searches and deletes a key, if it exists.
        pub fn delete(&mut self, k: &K) -> Option<(K, V)> {
            if let Ok(p) = self.as_spat_slice().search_point(&k) {
                Some(self.inner.remove(p))
            } else {
                None
            }
        }

        /// Removes element at index.
        ///
        /// Panics on invalid index.
        pub fn remove(&mut self, index: usize) -> (K, V) {
            self.inner.remove(index)
        }
    }

    impl<K, V> core::ops::Deref for SpatVec<K, V> {
        type Target = [(K, V)];
        fn deref(&self) -> &[(K, V)] {
            &*self.inner
        }
    }

    impl<'a, K, V> From<&'a SpatVec<K, V>> for SpatSlice<'a, K, V> {
        fn from(t: &'a SpatVec<K, V>) -> Self {
            Self { inner: &*t.inner }
        }
    }
}
