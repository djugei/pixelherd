#![allow(unused)]
/// a vec-like structure where removes do not disturb the indices of other elements
/// basically Vec<Option<T>>
#[derive(Default, Debug, Clone)]
pub struct StableVec<T> {
    inner: Vec<Option<T>>,
}

impl<T> StableVec<T> {
    pub fn with_capacity(c: usize) -> Self {
        Self {
            inner: Vec::with_capacity(c),
        }
    }
    /// pushes a new element after all the others
    /// this is amortized O(1)
    pub fn push(&mut self, e: T) {
        self.inner.push(Some(e))
    }
    /// adds a new element, trying to reuse a dead location
    /// before appending
    /// this is O(n)
    /// returns its position
    pub fn add(&mut self, e: T) -> usize {
        let empty = self
            .inner
            .iter_mut()
            .enumerate()
            .filter(|(_i, e)| e.is_none())
            .next();
        if let Some((i, empty)) = empty {
            *empty = Some(e);
            i
        } else {
            self.push(e);
            self.inner.len() - 1
        }
    }
    /// panics on oob
    /// returns the element if an element existed at the position
    /// if the position was already empty returns None
    pub fn remove(&mut self, index: usize) -> Option<T> {
        let e = &mut self.inner[index];
        let mut out = None;
        std::mem::swap(&mut out, e);
        out
    }

    /// panics on oob,
    /// returns refercence to element at position or none
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner[index].as_ref()
    }
    /// removes all empty entries
    /// this modifies the index of all entries after the first empty one
    /// you might want to call shrink_to_fit afterwards
    pub fn collapse(&mut self) {
        self.inner.retain(|e| e.is_some());
    }

    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    /// i don't wanna implement parallel iterator
    pub fn inner_mut(&mut self) -> &mut Vec<Option<T>> {
        &mut self.inner
    }
    pub fn inner(&self) -> &Vec<Option<T>> {
        &self.inner
    }

    pub fn iter_indexed(&self) -> impl Iterator<Item = (usize, &T)> {
        self.inner
            .iter()
            .enumerate()
            .map(|(i, e)| e.as_ref().map(|e| (i, e)))
            .flatten()
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        for e in self.inner.iter_mut() {
            let del = e
                .as_ref()
                .map(|e| !f(e))
                // no need to double-delete
                .unwrap_or(false);
            if del {
                *e = None
            }
        }
    }
    /// this is O(n)
    pub fn len(&self) -> usize {
        self.inner.iter().flatten().count()
    }
    /// iterator over all the empty slots
    /// every element of this will be None (i.e. .next() returns Some(None) or None)
    fn empty_iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut Option<T>)> {
        self.inner
            .iter_mut()
            .enumerate()
            .filter(|(_i, e)| e.is_none())
    }
}
impl<T> std::iter::Extend<T> for StableVec<T> {
    /// this is strictly more performant than calling add multiple times
    //todo: implement an extend-iter that -> impl Iterator<T=usize> with all the new positions
    fn extend<I>(&mut self, new: I)
    where
        I: IntoIterator<Item = T>,
    {
        let mut empties = self.inner.iter_mut().filter(|e| e.is_none());
        let mut new = new.into_iter();
        for empty in empties {
            let new = new.next();
            if new.is_some() {
                *empty = new
            } else {
                // exhausted input, we are done
                return;
            }
        }
        // exhausted empty slots, push to end
        for new in new {
            self.push(new)
        }
    }
}
