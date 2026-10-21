//! Regular {p,q} tiling via its (π/p, π/q, π/2) fundamental triangle with
//! vertices O (tile center), V (polygon vertex, on the x1 axis) and E (edge
//! midpoint, at angle π/p).

use std::f64::consts::PI;

use glam::DVec3;

use crate::{hyp::lor, quotient::Quotient};

/// Fold sweeps over the three mirrors; must match `MAX_SWEEPS` in tiling.wgsl.
pub const MAX_SWEEPS: u32 = 64;
/// Largest supported p and q. The fold needs more sweeps as p or q grows
/// (about 35 at 32); see `fold_converges_within_cap_at_rim`.
pub const MAX_PQ: u32 = 32;

/// The geometry a {p,q} tiling lives in, for p, q >= 3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Geometry {
    Spherical,
    Euclidean,
    Hyperbolic,
}

impl Geometry {
    pub fn of(p: u32, q: u32) -> Self {
        match ((p - 2) * (q - 2)).cmp(&4) {
            std::cmp::Ordering::Less => Self::Spherical,
            std::cmp::Ordering::Equal => Self::Euclidean,
            std::cmp::Ordering::Greater => Self::Hyperbolic,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Tiling {
    pub p: u32,
    pub q: u32,
    /// Inward unit normals of the mirrors OV, OE, VE. VE carries polygon edges.
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
    /// `None` unless {p,q} is hyperbolic and p, q <= MAX_PQ.
    pub fn new(p: u32, q: u32) -> Option<Self> {
        if !(3..=MAX_PQ).contains(&p)
            || !(3..=MAX_PQ).contains(&q)
            || Geometry::of(p, q) != Geometry::Hyperbolic
        {
            return None;
        }
        let (sp, cp) = (PI / p as f64).sin_cos();
        let (sq, cq) = (PI / q as f64).sin_cos();
        let cot_p = cp / sp;
        let cosh_r = cot_p / (PI / q as f64).tan();
        let k = ((cq * cq) / (sp * sp) - 1.0).sqrt();
        Some(Self {
            p,
            q,
            mirrors: [
                DVec3::new(0.0, 0.0, 1.0),
                DVec3::new(0.0, sp, -cp),
                DVec3::new(-k, -cq * cot_p, -cq),
            ],
            vert: DVec3::new(cosh_r, (cosh_r * cosh_r - 1.0).sqrt(), 0.0),
            half_edge: (cp / sq).acosh(),
            quotient: Quotient::first(p, q)?,
        })
    }

    /// Steps p by `dp` (or q by `dq`) repeatedly until {p,q} is valid;
    /// `None` if it leaves 3..=MAX_PQ first.
    pub fn stepped(&self, dp: i32, dq: i32) -> Option<Self> {
        let (mut p, mut q) = (self.p as i32, self.q as i32);
        loop {
            (p, q) = (p + dp, q + dq);
            if p < 3 || q < 3 {
                return None;
            }
            if let Some(t) = Self::new(p as u32, q as u32) {
                return Some(t);
            }
            if p as u32 > MAX_PQ || q as u32 > MAX_PQ {
                return None;
            }
        }
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
            for (k, n) in self.mirrors.into_iter().enumerate() {
                let d = lor(point, n);
                if d < 0.0 {
                    point -= 2.0 * d * n;
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
    use crate::hyp::{ORIGIN, disk_to_hyp, reflect, tests::assert_close, translate};
    use glam::DVec2;

    const PAIRS: [(u32, u32); 8] = [
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
    fn rejects_non_hyperbolic() {
        for (p, q) in [
            (3, 3),
            (4, 4),
            (6, 3),
            (3, 6),
            (4, 3),
            (2, 9),
            (9, 2),
            (3, 33),
        ] {
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
    fn stepping_skips_non_hyperbolic() {
        let pq = |t: Option<Tiling>| t.map(|t| (t.p, t.q));
        let t73 = Tiling::new(7, 3).unwrap();
        assert_eq!(pq(t73.stepped(1, 0)), Some((8, 3)));
        assert_eq!(pq(t73.stepped(-1, 0)), None);
        assert_eq!(pq(Tiling::new(4, 5).unwrap().stepped(0, -1)), None);
        assert_eq!(pq(Tiling::new(3, 8).unwrap().stepped(0, -1)), Some((3, 7)));
        assert_eq!(pq(Tiling::new(32, 3).unwrap().stepped(1, 0)), None);
    }

    #[test]
    fn gram_conditions() {
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let [a, b, c] = t.mirrors;
            let cos = |m: u32| (PI / m as f64).cos();
            for n in t.mirrors {
                assert!((lor(n, n) - 1.0).abs() < 1e-9);
            }
            assert!((lor(a, b) + cos(p)).abs() < 1e-9);
            assert!((lor(a, c) + cos(q)).abs() < 1e-9);
            assert!(lor(b, c).abs() < 1e-9);
            assert!((lor(t.vert, t.vert) + 1.0).abs() < 1e-9);
            assert!(lor(t.vert, a).abs() < 1e-9);
            assert!(lor(t.vert, c).abs() < 1e-9);
            // V's mirror image across OE is the polygon's next vertex.
            let v2 = crate::hyp::reflect(b) * t.vert;
            assert!((-lor(t.vert, v2) - (2.0 * t.half_edge).cosh()).abs() < 1e-9);
            // Origin strictly inside the triangle.
            assert!(lor(DVec3::X, c) > 0.0);
        }
    }

    #[test]
    fn fold_lands_in_triangle() {
        let t = Tiling::new(7, 3).unwrap();
        let x = disk_to_hyp(DVec2::new(0.6, -0.7));
        let f = t.fold(x);
        assert!(f.sweeps < MAX_SWEEPS);
        let y = f.point;
        assert!(t.mirrors.iter().all(|&n| lor(y, n) >= -1e-9));
        assert!((lor(y, y) + 1.0).abs() < 1e-9);
        let replayed = f.word.iter().fold(x, |x, &k| reflect(t.mirrors[k]) * x);
        assert_close(replayed, y, 1e-9);
    }

    /// Worst case of the shader's fold cap over the drawn region rho^2 <= 0.999,
    /// with the view centered on each corner of the fundamental triangle (where
    /// `View::settle` keeps it).
    #[test]
    fn fold_converges_within_cap_at_rim() {
        let rho = 0.999f64.sqrt();
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let [_, b, c] = t.mirrors.map(|n| DVec3::new(-n.x, n.y, n.z));
            let e = b.cross(c);
            let e = e / (-lor(e, e)).sqrt() * e.x.signum();
            let worst = [ORIGIN, t.vert, e]
                .into_iter()
                .flat_map(|center| {
                    let m_inv = translate(ORIGIN, center);
                    (0..4096).flat_map(move |i| (1..=8).map(move |r| (m_inv, i, r)))
                })
                .map(|(m_inv, i, r)| {
                    let a = i as f64 / 4096.0 * 2.0 * PI;
                    let d = rho * r as f64 / 8.0 * DVec2::from_angle(a);
                    t.fold(m_inv * disk_to_hyp(d)).sweeps
                })
                .max()
                .unwrap();
            println!("{{{p},{q}}}: worst {worst} sweeps");
            assert!(worst < MAX_SWEEPS, "{{{p},{q}}}: {worst}");
        }
    }
}
