use crate::app::TreeRef;
use crate::blip::Status;

#[derive(Clone, Copy, Debug)]
pub enum Selection {
    None,
    Bigboy,
    Age,
    Young,
    Spawns,
    Generation,
    Mouse,
}

impl Selection {
    pub fn new() -> Self {
        //Selection::None
        Selection::Mouse
    }
    // todo: add pre
    pub fn next(self) -> Self {
        match self {
            Selection::None => Selection::Bigboy,
            Selection::Bigboy => Selection::Age,
            Selection::Age => Selection::Young,
            Selection::Young => Selection::Spawns,
            Selection::Spawns => Selection::Generation,
            Selection::Generation => Selection::Mouse,
            Selection::Mouse => Selection::None,
        }
    }

    /// mousepos needs to be scaled to simulation coordinates already
    pub fn select<'a, I>(self, blips: I, tree: TreeRef, mousepos: &[f64; 2]) -> Option<usize>
    where
        I: Iterator<Item = (usize, &'a Status)>,
    {
        // i think this is where ppl use lenses?
        match self {
            Selection::None => None,
            Selection::Bigboy => blips
                .map(|(i, b)| (i, b.hp + b.food))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            Selection::Age => blips
                .map(|(i, b)| (i, b.age))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            Selection::Young => blips
                .map(|(i, b)| (i, b.age))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0),
            Selection::Spawns => blips
                .map(|(i, b)| (i, b.children))
                .max_by(|a, b| a.1.cmp(&b.1))
                .map(|c| c.0),
            Selection::Generation => blips
                .map(|(i, b)| (i, b.generation))
                .max_by(|a, b| a.1.cmp(&b.1))
                .map(|c| c.0),
            Selection::Mouse => {
                // the anti_r is not very good at this operation
                use crate::config;
                tree.query_distance(mousepos, config::b::LOCAL_ENV * 10.)
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|(_d, (_p, i))| *i)
            }
        }
    }
}
