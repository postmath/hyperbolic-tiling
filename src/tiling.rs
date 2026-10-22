//! Regular {p,q} tiling via its (π/p, π/q, π/2) fundamental triangle with
//! vertices O (tile center), V (polygon vertex) and E (edge midpoint). For
//! finite p, O is the origin and V is on the x1 axis. For p = ∞, O is ideal,
//! the center of a horocycle through the vertices of an apeirogon; E is then
//! the origin, with VE along the x1 axis and O straight up. For q = ∞, V is
//! ideal.

use std::{f64::consts::PI, fmt};

use glam::DVec3;

use crate::{geom::Geometry, quotient::Quotient};

/// Fold sweeps over the three mirrors; must match `MAX_SWEEPS` in tiling.wgsl.
pub const MAX_SWEEPS: u32 = 64;
/// Largest supported finite p and q. The fold needs more sweeps as p or q
/// grows (about 35 at 32); see `fold_converges_within_cap`.
pub const MAX_PQ: u32 = 32;
/// Cusp steps beyond this are skipped, as they are in the shader, where k
/// must fit an i32 and f32 has long lost the point anyway.
const MAX_CUSP_STEP: f64 = 1e7;
/// |OV| of Euclidean tilings, whose size is otherwise arbitrary. Close to that
/// of the hyperbolic {7,3} and the spherical {3,5} and {5,3}.
const EUCLIDEAN_RADIUS: f64 = 0.6;

/// p or q: sides per tile or tiles per vertex, 3..=MAX_PQ or infinite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Count {
    Finite(u32),
    Infinite,
}

impl Count {
    pub fn all() -> impl Iterator<Item = Self> {
        (3..=MAX_PQ).map(Self::Finite).chain([Self::Infinite])
    }

    /// π/m, the angle of the fundamental triangle at the corresponding
    /// corner: 0 where that corner is ideal.
    pub fn angle(self) -> f64 {
        match self {
            Self::Finite(m) => PI / m as f64,
            Self::Infinite => 0.0,
        }
    }

    /// The next count in `all` in direction `dir` (±1).
    pub fn stepped(self, dir: i32) -> Option<Self> {
        let all: Vec<_> = Self::all().collect();
        let i = all.iter().position(|&c| c == self)?;
        all.get(i.checked_add_signed(dir as isize)?).copied()
    }
}

impl From<u32> for Count {
    fn from(m: u32) -> Self {
        Self::Finite(m)
    }
}

impl fmt::Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Finite(m) => write!(f, "{m}"),
            Self::Infinite => write!(f, "∞"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Tiling {
    pub p: Count,
    pub q: Count,
    pub geometry: Geometry,
    /// Inward unit covectors of the mirrors OV, OE, VE. VE carries polygon edges.
    pub mirrors: [DVec3; 3],
    /// V, unless it is ideal.
    pub vert: Option<DVec3>,
    /// The length that caps lines and dots: half the polygon edge |VE|, or
    /// for q = ∞, where that is infinite, the distance from E to OV.
    pub size: f64,
    pub quotient: Quotient,
}

/// A line of the tiling as integer coefficients α of the mirrors: the line
/// with covector Σ α_m c_m.
pub type Line = [i64; 3];

/// Mirror k as a `Line`.
pub fn mirror(k: usize) -> Line {
    std::array::from_fn(|m| (m == k) as i64)
}

// `point` and `sweeps` are only read by tests that replay the shader.
#[cfg_attr(not(test), allow(dead_code))]
pub struct Fold {
    pub point: DVec3,
    /// The lines reflected in, in order: point = r_last ... r_first x.
    pub word: Vec<Line>,
    pub sweeps: u32,
}

impl Tiling {
    /// `None` unless p and q are in `Count::all`.
    pub fn new(p: impl Into<Count>, q: impl Into<Count>) -> Option<Self> {
        let (p, q) = (p.into(), q.into());
        if !Count::all().any(|c| c == p) || !Count::all().any(|c| c == q) {
            return None;
        }
        let g = Geometry::of(p, q);
        let (sp, cp) = p.angle().sin_cos();
        let cq = q.angle().cos();
        let mirrors = if p == Count::Infinite {
            // OV makes angle 0 with the x2 axis OE, and π/q with the x1
            // axis VE; its offset follows from the normalization.
            let a0 = (-g.kappa() * (cp * cp + cq * cq - 1.0)).sqrt();
            [DVec3::new(a0, -cp, -cq), DVec3::Y, DVec3::Z]
        } else {
            // VE is perpendicular to OE and meets OV at angle π/q; its offset
            // c0 follows from the normalization, except in the Euclidean case.
            let c1 = -cq * cp / sp;
            let c0 = match g {
                Geometry::Euclidean => -EUCLIDEAN_RADIUS * c1,
                _ => (g.kappa() * (1.0 - (cq / sp).powi(2))).sqrt(),
            };
            [DVec3::Z, DVec3::new(0.0, sp, -cp), DVec3::new(c0, c1, -cq)]
        };
        let vert = g.meet(mirrors[0], mirrors[2]);
        let size = match vert {
            Some(v) => 0.5 * g.dist(v, g.reflect(mirrors[1]) * v),
            None => g.asin(mirrors[0].dot(g.meet(mirrors[1], mirrors[2])?)),
        };
        Some(Self {
            p,
            q,
            geometry: g,
            mirrors,
            vert,
            size,
            quotient: Quotient::first(p, q)?,
        })
    }

    /// Steps p by `dp` (or q by `dq`) through `Count::all`; `None` if that
    /// leaves it.
    pub fn stepped(&self, dp: i32, dq: i32) -> Option<Self> {
        let step = |c: Count, d| if d == 0 { Some(c) } else { c.stepped(d) };
        Self::new(step(self.p, dp)?, step(self.q, dq)?)
    }

    /// Steps ℓ for the tile coloring to the next admissible prime in direction `dir`.
    pub fn with_ell_stepped(&self, dir: i64) -> Option<Self> {
        let quotient = Quotient::stepped(self.p, self.q, self.quotient.ell, dir)?;
        Some(Self { quotient, ..*self })
    }

    pub fn covector(&self, line: Line) -> DVec3 {
        (0..3).map(|m| line[m] as f64 * self.mirrors[m]).sum()
    }

    /// The mirrors j parallel to mirror 0, OV, meeting it at an ideal corner:
    /// OE if p = ∞, VE if q = ∞.
    pub fn cusps(&self) -> impl Iterator<Item = usize> {
        [(1, self.p), (2, self.q)]
            .into_iter()
            .filter(|&(_, m)| m == Count::Infinite)
            .map(|(j, _)| j)
    }

    /// The local width of the cusp between OV and mirror j at a point x of
    /// the fundamental triangle: the sum of sin_κ of the distances to the
    /// two, which is invariant under both reflections and goes to 0 deep in
    /// the cusp.
    pub fn cusp_width(&self, j: usize, x: DVec3) -> f64 {
        x.dot(self.mirrors[0] + self.mirrors[j])
    }

    /// Reflects `x` into the fundamental triangle, exactly as the shader does.
    ///
    /// Between OV and a parallel mirror j, the reflections act on
    /// u = c_0·x like the infinite dihedral group on a line, fixing the
    /// `cusp_width` h: the strip is 0 <= u <= h, and the translation
    /// σ_0 σ_j moves u by -2h. Rather than bounce there, the fold applies its
    /// k-th power at once, with k = u/2h rounded toward 0, which leaves
    /// |u| < 2h for at most two single reflections. The power is the
    /// reflection in the k-th image c_0 - k(c_0 + c_j) of OV followed by the
    /// one in OV; that image is a unit covector too, since c_0 + c_j is null.
    pub fn fold(&self, x: DVec3) -> Fold {
        let (mut point, mut word) = (x, vec![]);
        let mut reflect = |point: &mut DVec3, line: Line| {
            let c = self.covector(line);
            *point -= 2.0 * point.dot(c) * self.geometry.normal(c);
            word.push(line);
        };
        for sweeps in 0..MAX_SWEEPS {
            let mut changed = false;
            for j in self.cusps() {
                let (u, h) = (point.dot(self.mirrors[0]), self.cusp_width(j, point));
                if h <= 0.0 || u.abs() >= 2.0 * h * MAX_CUSP_STEP {
                    continue;
                }
                let k = (u / (2.0 * h)).trunc();
                if k != 0.0 {
                    let k = k as i64;
                    let mut image = [1 - k, 0, 0];
                    image[j] = -k;
                    reflect(&mut point, image);
                    reflect(&mut point, mirror(0));
                    changed = true;
                }
            }
            for (k, c) in self.mirrors.into_iter().enumerate() {
                if point.dot(c) < 0.0 {
                    reflect(&mut point, mirror(k));
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
    use crate::{
        geom::{
            ORIGIN,
            tests::{MU, assert_close},
        },
        quotient::{E_C, IDENTITY},
    };
    use Count::{Finite as F, Infinite as INF};
    use glam::DVec2;

    const PAIRS: [(Count, Count); 22] = [
        (F(3), F(3)),
        (F(3), F(4)),
        (F(4), F(3)),
        (F(3), F(5)),
        (F(5), F(3)),
        (F(4), F(4)),
        (F(3), F(6)),
        (F(6), F(3)),
        (F(7), F(3)),
        (F(3), F(7)),
        (F(5), F(4)),
        (F(4), F(5)),
        (F(8), F(8)),
        (F(3), F(32)),
        (F(32), F(3)),
        (F(32), F(32)),
        (F(3), INF),
        (INF, F(3)),
        (F(8), INF),
        (INF, F(8)),
        (F(32), INF),
        (INF, INF),
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
                .filter(|&(p, q)| Geometry::of(F(p), F(q)) == g)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            all(Geometry::Spherical),
            [(3, 3), (3, 4), (3, 5), (4, 3), (5, 3)]
        );
        assert_eq!(all(Geometry::Euclidean), [(3, 6), (4, 4), (6, 3)]);
        assert_eq!(Geometry::of(F(3), INF), Geometry::Hyperbolic);
        assert_eq!(Geometry::of(INF, INF), Geometry::Hyperbolic);
    }

    #[test]
    fn stepping_stays_in_range() {
        let pq = |t: Option<Tiling>| t.map(|t| (t.p, t.q));
        let t73 = Tiling::new(7, 3).unwrap();
        assert_eq!(pq(t73.stepped(1, 0)), Some((F(8), F(3))));
        assert_eq!(pq(t73.stepped(-1, 0)), Some((F(6), F(3))));
        assert_eq!(pq(t73.stepped(0, -1)), None);
        let t32 = Tiling::new(32, 3).unwrap();
        assert_eq!(pq(t32.stepped(1, 0)), Some((INF, F(3))));
        let t_inf = Tiling::new(INF, 3).unwrap();
        assert_eq!(pq(t_inf.stepped(1, 0)), None);
        assert_eq!(pq(t_inf.stepped(-1, 0)), Some((F(32), F(3))));
    }

    #[test]
    fn gram_conditions() {
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let g = t.geometry;
            let [a, b, c] = t.mirrors;
            for n in t.mirrors {
                assert!((g.form(n, n) - 1.0).abs() < 1e-9);
            }
            assert!((g.form(a, b) + p.angle().cos()).abs() < 1e-9);
            assert!((g.form(a, c) + q.angle().cos()).abs() < 1e-9);
            assert!(g.form(b, c).abs() < 1e-9);
            // Exactly the corners with angle 0 are ideal.
            let o = g.meet(a, b);
            assert_eq!(o.is_none(), p == INF);
            assert_eq!(t.vert.is_none(), q == INF);
            let e = g.meet(b, c).unwrap();
            // Each corner strictly inside the opposite mirror.
            assert!(a.dot(e) > 0.0);
            if let Some(o) = o {
                assert!(c.dot(o) > 0.0);
            }
            if let Some(v) = t.vert {
                assert!(b.dot(v) > 0.0);
                assert!(v.dot(a).abs() < 1e-9);
                assert!(v.dot(c).abs() < 1e-9);
            }
            // Right triangle VEO: cos_κ |VE| = cos(π/p) / sin(π/q), or in the
            // plane |VE| = |OV| sin(π/p).
            let (sp, sq) = (p.angle().sin(), q.angle().sin());
            let err = match g {
                _ if q == INF => 0.0,
                Geometry::Spherical => t.size.cos() - p.angle().cos() / sq,
                Geometry::Euclidean => t.size - EUCLIDEAN_RADIUS * sp,
                Geometry::Hyperbolic => t.size.cosh() - p.angle().cos() / sq,
            };
            assert!(err.abs() < 1e-9, "{{{p},{q}}}");
            assert!(t.size.is_finite() && t.size > 0.0, "{{{p},{q}}}");
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
            let replayed = (f.word.iter()).fold(x, |x, &line| g.reflect(t.covector(line)) * x);
            assert_close(replayed, y, 1e-9 * x.length());
        }
    }

    /// Worst case of the shader's fold cap over what can be on screen, with
    /// the view centered on each finite corner of the fundamental triangle
    /// (where `View::settle` keeps it): the picture up to 0.999 of its rim2,
    /// or without a rim the sphere up to near the antipode, and a plane 4 disk
    /// radii across from the center. The Poincaré disk reaches farthest.
    /// Near an ideal corner a plain fold would bounce between two parallel
    /// mirrors, about 2/(1 - rho^2) times in the Poincaré disk; the cusp steps
    /// replace that.
    #[test]
    fn fold_converges_within_cap() {
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let g = t.geometry;
            let [a, b, c] = t.mirrors;
            let corners: Vec<_> = [g.meet(a, b), t.vert, g.meet(b, c)]
                .into_iter()
                .flatten()
                .collect();
            for mu in [0.0, 0.5, 1.0] {
                let rho = match g.rim2(mu) {
                    rim2 if rim2.is_finite() => (0.999 * rim2).sqrt(),
                    _ if g == Geometry::Spherical => 100.0,
                    _ => 4.0,
                };
                let worst = corners
                    .iter()
                    .flat_map(|&center| {
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

    /// The cusp steps reach the same triangle and tile as reflecting one
    /// mirror at a time: same point, parity and color.
    #[test]
    fn cusp_steps_agree_with_single_reflections() {
        for (p, q) in [(F(3), INF), (INF, F(3)), (INF, INF)] {
            let t = Tiling::new(p, q).unwrap();
            let (g, quot) = (t.geometry, &t.quotient);
            let color = |word: &[Line]| {
                let a = (word.iter()).fold(IDENTITY, |a, &line| quot.reflect_right(a, line));
                quot.apply(a, E_C)
            };
            for i in 0..64 {
                let d = 0.99 * DVec2::from_angle(i as f64 * 0.1);
                let x = g.disk_to_point(d, MU);
                let f = t.fold(x);
                // The same, one mirror at a time.
                let (mut y, mut word) = (x, vec![]);
                while let Some(k) = (0..3).find(|&k| y.dot(t.mirrors[k]) < 0.0) {
                    y = g.reflect(t.mirrors[k]) * y;
                    word.push(mirror(k));
                }
                assert_close(f.point, y, 1e-6);
                assert_eq!(f.word.len() % 2, word.len() % 2);
                assert_eq!(color(&f.word), color(&word), "{{{p},{q}}} {d}");
            }
        }
    }

    /// Points where the cusp width at an ideal O is below its value at E lie
    /// in the apeirogon of the fundamental triangle: their fold reflects only
    /// in lines through O. The shader relies on this where f32 loses them.
    #[test]
    fn deep_in_an_ideal_center_is_the_central_apeirogon() {
        for q in [F(3), F(27), INF] {
            let t = Tiling::new(INF, q).unwrap();
            let g = t.geometry;
            let e = g.meet(t.mirrors[1], t.mirrors[2]).unwrap();
            let h_e = t.cusp_width(1, e);
            let mut inside = 0;
            for i in 0..4096 {
                let rho = 1.0 - 0.5f64.powi(i % 12 + 1);
                let d = rho * DVec2::from_angle(i as f64 * 0.0137);
                let x = g.disk_to_point(d, MU);
                if t.cusp_width(1, x) < h_e {
                    let f = t.fold(x);
                    assert!(f.word.iter().all(|line| line[2] == 0), "{q} at {d}");
                    inside += 1;
                }
            }
            assert!(inside > 100, "{q}: {inside}");
        }
    }
}
