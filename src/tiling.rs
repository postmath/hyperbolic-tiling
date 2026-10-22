//! Regular {p,q} tiling via its (π/p, π/q, π/2) fundamental triangle with
//! vertices O (tile center), V (polygon vertex, on the x1 axis) and E (edge
//! midpoint, at angle π/p).

use std::f64::consts::PI;

use glam::DVec3;

use crate::{geom::Geometry, quotient::Quotient};

/// Fold sweeps over the three mirrors; must match `MAX_SWEEPS` in tiling.wgsl.
pub const MAX_SWEEPS: u32 = 64;
/// Largest supported p and q. The fold needs more sweeps as p or q grows
/// (about 35 at 32); see `fold_converges_within_cap`.
pub const MAX_PQ: u32 = 32;
/// |OV| of Euclidean tilings, whose size is otherwise arbitrary. Close to that
/// of the hyperbolic {7,3} and the spherical {3,5} and {5,3}.
const EUCLIDEAN_RADIUS: f64 = 0.6;

#[derive(Clone, Copy, Debug)]
pub struct Tiling {
    pub p: u32,
    pub q: u32,
    pub geometry: Geometry,
    /// Inward unit covectors of the mirrors OV, OE, VE. VE carries polygon edges.
    pub mirrors: [DVec3; 3],
    pub vert: DVec3,
    /// Half the polygon edge length, |VE|.
    pub half_edge: f64,
    pub quotient: Quotient,
}

// `point` and `sweeps` are only read by tests that replay the shader.
#[cfg_attr(not(test), allow(dead_code))]
pub struct Fold {
    pub point: DVec3,
    /// Mirror indices in the order applied: point = r_last ... r_first x.
    pub word: Vec<usize>,
    pub sweeps: u32,
}

impl Tiling {
    /// `None` unless 3 <= p, q <= MAX_PQ.
    pub fn new(p: u32, q: u32) -> Option<Self> {
        if !(3..=MAX_PQ).contains(&p) || !(3..=MAX_PQ).contains(&q) {
            return None;
        }
        let g = Geometry::of(p, q);
        let (sp, cp) = (PI / p as f64).sin_cos();
        let cq = (PI / q as f64).cos();
        // VE is perpendicular to OE and meets OV at angle π/q; its offset c0
        // follows from the normalization, except in the Euclidean case.
        let c1 = -cq * cp / sp;
        let c0 = match g {
            Geometry::Euclidean => -EUCLIDEAN_RADIUS * c1,
            _ => (g.kappa() * (1.0 - (cq / sp).powi(2))).sqrt(),
        };
        let mirrors = [DVec3::Z, DVec3::new(0.0, sp, -cp), DVec3::new(c0, c1, -cq)];
        let vert = g.meet(mirrors[0], mirrors[2]);
        Some(Self {
            p,
            q,
            geometry: g,
            mirrors,
            vert,
            half_edge: 0.5 * g.dist(vert, g.reflect(mirrors[1]) * vert),
            quotient: Quotient::first(p, q)?,
        })
    }

    /// Steps p by `dp` (or q by `dq`); `None` if that leaves 3..=MAX_PQ.
    pub fn stepped(&self, dp: i32, dq: i32) -> Option<Self> {
        let p = self.p.checked_add_signed(dp)?;
        let q = self.q.checked_add_signed(dq)?;
        Self::new(p, q)
    }

    /// Steps ℓ for the tile coloring to the next admissible prime in direction `dir`.
    pub fn with_ell_stepped(&self, dir: i64) -> Option<Self> {
        let quotient = Quotient::stepped(self.p, self.q, self.quotient.ell, dir)?;
        Some(Self { quotient, ..*self })
    }

    /// Reflects `x` into the fundamental triangle, exactly as the shader does.
    pub fn fold(&self, x: DVec3) -> Fold {
        let (mut point, mut word) = (x, vec![]);
        for sweeps in 0..MAX_SWEEPS {
            let mut changed = false;
            for (k, c) in self.mirrors.into_iter().enumerate() {
                let d = point.dot(c);
                if d < 0.0 {
                    point -= 2.0 * d * self.geometry.normal(c);
                    word.push(k);
                    changed = true;
                }
            }
            if !changed {
                return Fold {
                    point,
                    word,
                    sweeps,
                };
            }
        }
        Fold {
            point,
            word,
            sweeps: MAX_SWEEPS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{
        ORIGIN,
        tests::{MU, assert_close},
    };
    use glam::DVec2;

    const PAIRS: [(u32, u32); 16] = [
        (3, 3),
        (3, 4),
        (4, 3),
        (3, 5),
        (5, 3),
        (4, 4),
        (3, 6),
        (6, 3),
        (7, 3),
        (3, 7),
        (5, 4),
        (4, 5),
        (8, 8),
        (3, 32),
        (32, 3),
        (32, 32),
    ];

    #[test]
    fn rejects_out_of_range() {
        for (p, q) in [(2, 9), (9, 2), (3, 33), (33, 3)] {
            assert!(Tiling::new(p, q).is_none(), "{{{p},{q}}}");
        }
    }

    #[test]
    fn geometry_classifies_the_small_cases() {
        let all = |g| {
            (3..=MAX_PQ)
                .flat_map(|p| (3..=MAX_PQ).map(move |q| (p, q)))
                .filter(|&(p, q)| Geometry::of(p, q) == g)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            all(Geometry::Spherical),
            [(3, 3), (3, 4), (3, 5), (4, 3), (5, 3)]
        );
        assert_eq!(all(Geometry::Euclidean), [(3, 6), (4, 4), (6, 3)]);
    }

    #[test]
    fn stepping_stays_in_range() {
        let pq = |t: Option<Tiling>| t.map(|t| (t.p, t.q));
        let t73 = Tiling::new(7, 3).unwrap();
        assert_eq!(pq(t73.stepped(1, 0)), Some((8, 3)));
        assert_eq!(pq(t73.stepped(-1, 0)), Some((6, 3)));
        assert_eq!(pq(t73.stepped(0, -1)), None);
        assert_eq!(pq(Tiling::new(32, 3).unwrap().stepped(1, 0)), None);
    }

    #[test]
    fn gram_conditions() {
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let g = t.geometry;
            let [a, b, c] = t.mirrors;
            let cos = |m: u32| (PI / m as f64).cos();
            for n in t.mirrors {
                assert!((g.form(n, n) - 1.0).abs() < 1e-9);
            }
            assert!((g.form(a, b) + cos(p)).abs() < 1e-9);
            assert!((g.form(a, c) + cos(q)).abs() < 1e-9);
            assert!(g.form(b, c).abs() < 1e-9);
            let v = t.vert;
            assert!((v.x * v.x + g.kappa() * (v.y * v.y + v.z * v.z) - 1.0).abs() < 1e-9);
            assert!(v.dot(a).abs() < 1e-9);
            assert!(v.dot(c).abs() < 1e-9);
            // Right triangle VEO: cos_κ |VE| = cos(π/p) / sin(π/q), or in the
            // plane |VE| = |OV| sin(π/p).
            let (sp, sq) = ((PI / p as f64).sin(), (PI / q as f64).sin());
            let err = match g {
                Geometry::Spherical => t.half_edge.cos() - cos(p) / sq,
                Geometry::Euclidean => t.half_edge - EUCLIDEAN_RADIUS * sp,
                Geometry::Hyperbolic => t.half_edge.cosh() - cos(p) / sq,
            };
            assert!(err.abs() < 1e-9, "{{{p},{q}}}");
            // Origin strictly inside the triangle.
            assert!(ORIGIN.dot(c) > 0.0);
        }
    }

    #[test]
    fn fold_lands_in_triangle() {
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let g = t.geometry;
            let x = g.disk_to_point(DVec2::new(0.6, -0.7), MU);
            let f = t.fold(x);
            assert!(f.sweeps < MAX_SWEEPS);
            let y = f.point;
            assert!(t.mirrors.iter().all(|&c| y.dot(c) >= -1e-9));
            let replayed = f.word.iter().fold(x, |x, &k| g.reflect(t.mirrors[k]) * x);
            assert_close(replayed, y, 1e-9);
        }
    }

    /// Worst case of the shader's fold cap over what can be on screen, with
    /// the view centered on each corner of the fundamental triangle (where
    /// `View::settle` keeps it): the picture up to 0.999 of its rim2, or
    /// without a rim the sphere up to near the antipode, and a plane 4 disk
    /// radii across from the center. The Poincaré disk reaches farthest.
    #[test]
    fn fold_converges_within_cap() {
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let g = t.geometry;
            let e = g.meet(t.mirrors[1], t.mirrors[2]);
            for mu in [0.0, 0.5, 1.0] {
                let rho = match g.rim2(mu) {
                    rim2 if rim2.is_finite() => (0.999 * rim2).sqrt(),
                    _ if g == Geometry::Spherical => 100.0,
                    _ => 4.0,
                };
                let worst = [ORIGIN, t.vert, e]
                    .into_iter()
                    .flat_map(|center| {
                        let m_inv = g.translate(ORIGIN, center);
                        (0..4096).flat_map(move |i| (1..=8).map(move |r| (m_inv, i, r)))
                    })
                    .map(|(m_inv, i, r)| {
                        let a = i as f64 / 4096.0 * 2.0 * PI;
                        let d = rho * r as f64 / 8.0 * DVec2::from_angle(a);
                        t.fold(m_inv * g.disk_to_point(d, mu)).sweeps
                    })
                    .max()
                    .unwrap();
                println!("{{{p},{q}}} μ = {mu}: worst {worst} sweeps");
                assert!(worst < MAX_SWEEPS, "{{{p},{q}}} μ = {mu}: {worst}");
            }
        }
    }
}
