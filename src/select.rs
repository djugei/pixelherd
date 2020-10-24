use crate::app::TreeRef;
use crate::blip::{Genes, Status};
use crate::brains::Brain;

#[derive(Clone, Copy, Debug)]
pub enum Selection {
    None,
    Bigboy,
    Age,
    Young,
    Spawns,
    Generation,
    Mouse,
    Lineage,
    Vore,
    AntiVore,
}

impl Selection {
    pub fn new() -> Self {
        //Selection::None
        Selection::Mouse
    }
    // todo: add pre
    pub fn rotate(self) -> Self {
        match self {
            Selection::None => Selection::Bigboy,
            Selection::Bigboy => Selection::Age,
            Selection::Age => Selection::Young,
            Selection::Young => Selection::Spawns,
            Selection::Spawns => Selection::Generation,
            Selection::Generation => Selection::Mouse,
            Selection::Mouse => Selection::Lineage,
            Selection::Lineage => Selection::Vore,
            Selection::Vore => Selection::AntiVore,
            Selection::AntiVore => Selection::None,
        }
    }
    pub fn rotate_rev(self) -> Self {
        match self {
            Selection::Bigboy => Selection::None,
            Selection::Age => Selection::Bigboy,
            Selection::Young => Selection::Age,
            Selection::Spawns => Selection::Young,
            Selection::Generation => Selection::Spawns,
            Selection::Mouse => Selection::Generation,
            Selection::Lineage => Selection::Mouse,
            Selection::Vore => Selection::Lineage,
            Selection::AntiVore => Selection::Vore,
            Selection::None => Selection::AntiVore,
        }
    }

    /// mousepos needs to be scaled to simulation coordinates already
    pub fn select<'a, I, B>(self, blips: I, tree: TreeRef, mousepos: &[f64; 2]) -> Option<usize>
    where
        I: Iterator<Item = (usize, (&'a Status, &'a Genes<B>))>,
        B: Brain + 'a,
    {
        match self {
            Selection::None => None,
            Selection::Bigboy => blips
                .map(|(i, (s, _g))| (i, s.hp + s.food))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            Selection::Age => blips
                .map(|(i, (s, _g))| (i, s.age))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            Selection::Young => blips
                .map(|(i, (s, _g))| (i, s.age))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            Selection::Spawns => blips
                .map(|(i, (s, _g))| (i, s.children))
                .max_by(|a, b| a.1.cmp(&b.1))
                .map(|c| c.0),
            Selection::Generation => blips
                .map(|(i, (s, _g))| (i, s.generation))
                .max_by(|a, b| a.1.cmp(&b.1))
                .map(|c| c.0),
            Selection::Mouse => {
                // the anti_r is not very good at this operation
                use crate::config;
                tree.query_distance(mousepos, config::b::LOCAL_ENV * 10.)
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|(_d, (_p, i))| *i)
            }
            Selection::Lineage => blips
                .map(|(i, (s, _g))| (i, s.lineage))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            Selection::Vore => blips
                .map(|(i, (_s, g))| (i, g.vore))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            Selection::AntiVore => blips
                .map(|(i, (_s, g))| (i, g.vore))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
        }
    }
}

impl Iterator for Selection {
    type Item = Self;
    fn next(&mut self) -> Option<Self> {
        *self = self.rotate();
        Some(*self)
    }
}

impl DoubleEndedIterator for Selection {
    fn next_back(&mut self) -> Option<Self> {
        *self = self.rotate_rev();
        Some(*self)
    }
}
