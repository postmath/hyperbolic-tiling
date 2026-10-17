//! Regular {p,q} tiling via its (π/p, π/q, π/2) fundamental triangle with
//! vertices O (tile center), V (polygon vertex, on the x1 axis) and E (edge
//! midpoint, at angle π/p).

use std::f64::consts::PI;

use glam::{DMat3, DVec3};

use crate::hyp::{lor, reflect};

/// Fold sweeps over the three mirrors; must match `MAX_SWEEPS` in tiling.wgsl.
pub const MAX_SWEEPS: u32 = 64;
/// Largest supported p and q. The fold needs more sweeps as p or q grows
/// (about 35 at 32); see `fold_converges_within_cap_at_rim`.
pub const MAX_PQ: u32 = 32;

#[derive(Clone, Copy, Debug)]
pub struct Tiling {
    pub p: u32,
    pub q: u32,
    /// Inward unit normals of the mirrors OV, OE, VE. VE carries polygon edges.
    pub mirrors: [DVec3; 3],
    pub vert: DVec3,
    /// Half the polygon edge length, |VE|.
    pub half_edge: f64,
}

impl Tiling {
    /// `None` unless {p,q} is hyperbolic, i.e. (p-2)(q-2) > 4, and p, q <= MAX_PQ.
    pub fn new(p: u32, q: u32) -> Option<Self> {
        if !(3..=MAX_PQ).contains(&p) || !(3..=MAX_PQ).contains(&q) || (p - 2) * (q - 2) <= 4 {
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
        })
    }

    /// Reflects `x` into the fundamental triangle, exactly as the shader does.
    /// Returns the folded point, the product F of the reflections applied
    /// (F x = folded), their count, and the number of sweeps used.
    pub fn fold(&self, x: DVec3) -> (DVec3, DMat3, u32, u32) {
        let (mut t, mut f, mut count) = (x, DMat3::IDENTITY, 0);
        for sweep in 0..MAX_SWEEPS {
            let mut changed = false;
            for n in self.mirrors {
                let d = lor(t, n);
                if d < 0.0 {
                    t -= 2.0 * d * n;
                    f = reflect(n) * f;
                    count += 1;
                    changed = true;
                }
            }
            if !changed {
                return (t, f, count, sweep);
            }
        }
        (t, f, count, MAX_SWEEPS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyp::{disk_to_hyp, tests::assert_close};
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
        let (y, f, _, sweeps) = t.fold(x);
        assert!(sweeps < MAX_SWEEPS);
        assert!(t.mirrors.iter().all(|&n| lor(y, n) >= -1e-9));
        assert!((lor(y, y) + 1.0).abs() < 1e-9);
        assert_close(f * x, y, 1e-9);
    }

    /// Worst case of the shader's fold cap over the drawn region rho^2 <= 0.999.
    #[test]
    fn fold_converges_within_cap_at_rim() {
        let rho = 0.999f64.sqrt();
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let worst = (0..4096)
                .flat_map(|i| (1..=8).map(move |r| (i, r)))
                .map(|(i, r)| {
                    let a = i as f64 / 4096.0 * 2.0 * PI;
                    let d = rho * r as f64 / 8.0 * DVec2::from_angle(a);
                    t.fold(disk_to_hyp(d)).3
                })
                .max()
                .unwrap();
            println!("{{{p},{q}}}: worst {worst} sweeps");
            assert!(worst < MAX_SWEEPS, "{{{p},{q}}}: {worst}");
        }
    }
}
