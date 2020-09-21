use crate::brains::Brain;
use crate::Blip;
use crate::BlipLoc;
use rstar::RTree;

#[derive(Clone, Copy, Debug)]
pub struct Selection(SelectionE);
impl Selection {
    pub fn new() -> Self {
        Self(SelectionE::new())
    }
    pub fn next(self) -> Self {
        Self(self.0.next())
    }
    /// mousepos needs to be scaled to simulation coordinates already
    pub fn select<B: Brain>(
        self,
        blips: &[Blip<B>],
        tree: &RTree<BlipLoc>,
        mousepos: &[f64; 2],
    ) -> Option<usize> {
        self.0.select(blips, tree, mousepos)
    }
}

#[derive(Clone, Copy, Debug)]
enum SelectionE {
    None,
    Bigboy,
    Age,
    Young,
    Spawns,
    Generation,
    Mouse,
}

impl SelectionE {
    fn new() -> Self {
        SelectionE::None
    }
    // todo: add pre
    fn next(self) -> Self {
        match self {
            SelectionE::None => SelectionE::Bigboy,
            SelectionE::Bigboy => SelectionE::Age,
            SelectionE::Age => SelectionE::Young,
            SelectionE::Young => SelectionE::Spawns,
            SelectionE::Spawns => SelectionE::Generation,
            SelectionE::Generation => SelectionE::Mouse,
            SelectionE::Mouse => SelectionE::None,
        }
    }

    /// mousepos needs to be scaled to simulation coordinates already
    fn select<B: Brain>(
        self,
        blips: &[Blip<B>],
        tree: &RTree<BlipLoc>,
        mousepos: &[f64; 2],
    ) -> Option<usize> {
        match self {
            SelectionE::None => None,
            SelectionE::Bigboy => blips
                .iter()
                .enumerate()
                .map(|(i, b)| (i, b.status.hp + b.status.food))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            SelectionE::Age => blips
                .iter()
                .enumerate()
                .map(|(i, b)| (i, b.status.age))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            SelectionE::Young => blips
                .iter()
                .enumerate()
                .map(|(i, b)| (i, b.status.age))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            SelectionE::Spawns => blips
                .iter()
                .enumerate()
                .map(|(i, b)| (i, b.status.children))
                .max_by(|a, b| a.1.cmp(&b.1))
                .map(|c| c.0),
            SelectionE::Generation => blips
                .iter()
                .enumerate()
                .map(|(i, b)| (i, b.status.generation))
                .max_by(|a, b| a.1.cmp(&b.1))
                .map(|c| c.0),
            SelectionE::Mouse => tree.nearest_neighbor(mousepos).map(|r| r.data),
        }
    }
}
