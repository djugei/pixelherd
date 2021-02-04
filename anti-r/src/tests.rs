extern crate std;
use crate::SpatVec;
use quickcheck::{Arbitrary, Gen};
use quickcheck_macros::quickcheck;
use std::boxed::Box;
use std::cmp;
use std::collections::HashSet;
use std::vec::Vec;

#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct Point(i32, i32);
#[derive(Clone, Copy, Debug)]
struct AABBRange(Point, Point);

impl Arbitrary for Point {
    fn arbitrary(g: &mut Gen) -> Self {
        Point(i32::arbitrary(g), i32::arbitrary(g))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let Point(x, y) = *self;
        Box::new(
            x.shrink()
                .map(move |new_x| Point(new_x, y))
                .chain(y.shrink().map(move |new_y| Point(x, new_y))),
        )
    }
}

impl Arbitrary for AABBRange {
    fn arbitrary(g: &mut Gen) -> Self {
        let xs = (i32::arbitrary(g), i32::arbitrary(g));
        let ys = (i32::arbitrary(g), i32::arbitrary(g));

        AABBRange(
            Point(cmp::min(xs.0, xs.1), cmp::min(ys.0, ys.1)),
            Point(cmp::max(xs.0, xs.1), cmp::max(ys.0, ys.1)),
        )
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let AABBRange(least, greatest) = *self;
        Box::new(
            least
                .shrink()
                .filter(move |&new_least| new_least.is_before_or_equal_on_all_coordinates(greatest))
                .map(move |new_least| AABBRange(new_least, greatest))
                .chain(
                    greatest
                        .shrink()
                        .filter(move |&new_greatest| {
                            new_greatest.is_after_or_equal_on_all_coordinates(least)
                        })
                        .map(move |new_greatest| AABBRange(least, new_greatest)),
                ),
        )
    }
}

impl Point {
    fn is_before_or_equal_on_all_coordinates(self, other: Point) -> bool {
        self.0 <= other.0 && self.1 <= other.1
    }

    fn is_after_or_equal_on_all_coordinates(self, other: Point) -> bool {
        self.0 >= other.0 && self.1 >= other.1
    }
}

fn linearly_filtered(
    points: Vec<(Point, ())>,
    range: AABBRange,
) -> impl Iterator<Item = (Point, ())> {
    let AABBRange(min_point, max_point) = range;
    points.into_iter().filter(move |&(to_check, ())| {
        to_check.is_after_or_equal_on_all_coordinates(min_point)
            && to_check.is_before_or_equal_on_all_coordinates(max_point)
    })
}

fn aabb_results(points: Vec<(Point, ())>, range: AABBRange) -> impl Iterator<Item = (Point, ())> {
    let AABBRange(min_point, max_point) = range;
    let spat_vec = SpatVec::new_from(points.clone());
    let (aabb_start, aabb_end) = spat_vec.as_spat_slice().query_aabb(&min_point, &max_point);
    (&spat_vec[aabb_start..aabb_end])
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
}

#[quickcheck]
/// This checks that our aabb search includes every point that a linear filtering would.
fn test_aabb_includes_every_point_from_linear_filtering(
    points: Vec<(Point, ())>,
    range: AABBRange,
) -> bool {
    let selected_points: HashSet<_> = aabb_results(points.clone(), range).collect();

    linearly_filtered(points, range)
        .all(|should_be_included| selected_points.contains(&should_be_included))
}

#[quickcheck]
/// This checks that our aabb search doesn't return any extra points that a linear filtering wouldn't.
fn test_aabb_only_includes_points_from_linear_filtering(
    points: Vec<(Point, ())>,
    range: AABBRange,
) -> bool {
    let filtered: HashSet<_> = linearly_filtered(points.clone(), range).collect();

    aabb_results(points.clone(), range)
        .all(|actually_included| filtered.contains(&actually_included))
}
