# Anti-R

<!-- cargo-sync-readme start -->

Anti-R contains a alternative spatial data structure that outperforms R-Trees for low amounts of nodes.

# Performance:
R-Trees and anti-r have the same O(n) complexity for all operations,
log(n) for searching and updating, n\*log(n) for creation.

They only differ by constant factors,
either x or y in O(log\_b(n+x)+y)
and the base of the logarithm,
which is 2 for Anti-R and configurable for R-Tree, generally 3-6.

Anti-R is always faster at updating all elements and bulk-loading by a constant factor,
therefore it is more noticeable for small n.

Full updates and bulk-loads are equivalent in speed for Anti-R.
For R-Trees full updates are never worth it,
a full reconstruction is simply faster.

Zero to a bit more than 10\_000 elements are faster to bounding box-query for Anti-R.
Zero to about 1000 elements are faster to distance-query for Anti-R.

R-Trees might be catching up quicker if the elements are weirdly distributed.

See the bench directory and the output of cargo bench (target/criterion) for more details.

Notice that this has been benched against the rstar crate,
which might not be the fastest implementation of an R-Tree in existence.
The benchmark results are exactly as expected though.

<!-- cargo-sync-readme end -->

# Contributing
See README.md at the root of the repository.
