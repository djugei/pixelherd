#![no_std]

/// a spatial slice, basically just a slice sorted by k.
/// if k is a position in space this means its sorted by its coordinates
/// which allows for some efficient queries
///
/// if you want to modify elements just let this go out of scope, modify the base slice and rebuild
/// the spatslice, potentially using new_unchecked if you uphold ordering
///
/// the trait bound on the K for most of the functions is PartialOrd.
/// this is a lie, if a comparison fails the function will panic.
/// therefore the "real" trait bound would be Ord.
/// its just insanely inconvenient to have to deal with that for floats which are probably the
/// mainly used datatype for this
#[derive(PartialEq, Debug)]
pub struct SpatSlice<'a, K, V> {
    inner: &'a [(K, V)],
}

impl<'a, K, V> SpatSlice<'a, K, V>
where
    K: PartialOrd,
{
    pub fn new(base: &'a mut [(K, V)]) -> Self {
        base.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self::new_unchecked(base)
    }
    /// when calling this instead of new you guarantee that the base is already sorted by K
    /// an unsorted base will result in logic errors, not unsafety
    pub fn new_unchecked(base: &'a [(K, V)]) -> Self {
        Self { inner: base }
    }

    /// searches for all entries that are enclosed between the edges min and max
    /// min must be strictly smaller on all coordinates than max for this to work
    /// if you want to query for entities enclosed by arbitrary points
    /// do an element-wise swap
    ///  [-1, -2, 3 ] and [1, 2, -3] would become [-1, -2, -3] and [1, 2, 3]
    pub fn querry_aabb(&self, min: &K, max: &K) -> (usize, usize) {
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

    /// this is literally just a binary search, as in slice.binary_search
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
    /// this is only implemented for 2d vectors rn, cause did not want to create a Distance and
    /// scalarsub/add trait.
    /// maybe there is one from a math library that i can use.
    /// this function is very simple though so you can easily implement it yourself if you need a
    /// different K
    ///
    /// dist2 is the squared distance, i.e. if you input 100 it will search for things in range 10.
    /// if you want to search for range 100 you have to input 100*100
    ///
    /// excludes the point itself from the query if an exact match exists (distance will never be
    /// 0)
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
        let (start, end) = self.querry_aabb(&min, &max);
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
pub mod vec {
    extern crate alloc;
    use crate::SpatSlice;
    use alloc::vec::Vec;

    #[derive(PartialEq, Debug)]
    pub struct SpatVec<K, V> {
        inner: Vec<(K, V)>,
    }

    impl<K, V> SpatVec<K, V>
    where
        K: PartialOrd,
    {
        pub fn new_from(mut v: Vec<(K, V)>) -> Self {
            SpatSlice::new(&mut v);
            Self::new_from_unchecked(v)
        }
        pub fn new_from_unchecked(v: Vec<(K, V)>) -> Self {
            Self { inner: v }
        }
        pub fn inner(self) -> Vec<(K, V)> {
            self.inner
        }
        /// applies f and then re-sorts
        pub fn map<F>(&mut self, f: F)
        where
            F: FnMut(&mut (K, V)),
        {
            self.inner.iter_mut().for_each(f);
            SpatSlice::new(&mut self.inner);
        }
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
        pub fn insert(&mut self, kv: (K, V)) {
            let p = crate::either(self.as_spat_slice().search_point(&kv.0));
            self.inner.insert(p, kv);
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
