pub const FOOD_WIDTH: usize = 50;
pub const FOOD_HEIGHT: usize = 50;

pub const INITIAL_CELLS: usize = 100;
pub const REPLACEMENT: usize = 5;

pub const SIM_WIDTH: f64 = (FOOD_WIDTH * 10) as f64;
pub const SIM_HEIGHT: f64 = (FOOD_HEIGHT * 10) as f64;

pub mod b {
    pub const LOCAL_ENV: f64 = 1500.;
    pub const MAX_SPEED: f64 = 10.;
    // how much energy is used by just existing vs by moving
    pub const IDLE_E_RATIO: f64 = 0.3;
    // how much energy is used by a blip at max speed/second
    pub const E_DRAIN: f64 = 1.;
}

// how much food should on average be created per second
// replenish / e_drain = number of max supported moving blips
// replenish / (e_drain * idle_e_ratio) = number of supported idle blips
// should probably be higher than replacement, or maybe not?
pub const REPLENISH: f64 = 25.;
