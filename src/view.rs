//! The view isometry and the drags that change it.

use glam::{DMat3, DVec2};

use crate::{
    geom::{Geometry, ORIGIN, rotate},
    quotient::{self, ModMat},
    tiling::Tiling,
};

/// Pan drags hold points up to this fraction of the picture's rim2, where the
/// shader starts its rim fade. Without a rim the whole screen is pannable.
const PAN_RIM2_FRACTION: f64 = 0.98;
/// Rotate drags ignore pointers this close to the center, where the angle is noise.
const ROTATE_RHO_MIN: f64 = 0.02;

pub struct View {
    pub geometry: Geometry,
    /// Tiling frame -> screen frame.
    pub m: DMat3,
    /// φ(G) for the symmetry G that `settle` has absorbed into `m`: a point x
    /// of the current tiling frame is G x in the frame the colors refer to.
    pub color_frame: ModMat,
}

impl View {
    pub fn new(geometry: Geometry) -> Self {
        Self {
            geometry,
            m: DMat3::IDENTITY,
            color_frame: quotient::IDENTITY,
        }
    }

    /// Makes the current tiling frame the one colors refer to; needed when the
    /// quotient changes.
    pub fn reset_colors(&mut self) {
        self.color_frame = quotient::IDENTITY;
    }

    fn pan_rho2_max(&self, mu: f64) -> f64 {
        PAN_RIM2_FRACTION * self.geometry.rim2(mu)
    }

    /// Whether a pan drag can hold the point at disk position `d`, in the
    /// projection with parameter `mu` (see `Geometry::disk_to_point`).
    pub fn pannable(&self, d: DVec2, mu: f64) -> bool {
        d.length_squared() <= self.pan_rho2_max(mu)
    }

    /// Moves the picture so the point under disk position `from` ends up under
    /// `to`. Points beyond `pannable` are clamped to its edge.
    pub fn pan(&mut self, from: DVec2, to: DVec2, mu: f64) {
        let clamp = |d: DVec2| {
            if self.pannable(d, mu) {
                d
            } else {
                d.normalize() * self.pan_rho2_max(mu).sqrt()
            }
        };
        let g = self.geometry;
        let (p, q) = (
            g.disk_to_point(clamp(from), mu),
            g.disk_to_point(clamp(to), mu),
        );
        self.m = g.translate(p, q) * self.m;
    }

    /// Rotates about the screen center by the change in the pointer's angle.
    pub fn rotate(&mut self, from: DVec2, to: DVec2) {
        if from.length() < ROTATE_RHO_MIN || to.length() < ROTATE_RHO_MIN {
            return;
        }
        self.m = rotate(from.angle_to(to)) * self.m;
    }

    /// Re-expresses the same picture with the view center folded into the
    /// fundamental triangle, so M stays near the identity however far one pans,
    /// then removes rounding drift from M.
    /// With F = r_last ... r_first the fold, M becomes M F⁻¹ = M r_first ... r_last.
    pub fn settle(&mut self, tiling: &Tiling) {
        let g = self.geometry;
        let mut word = tiling.fold(self.m.inverse() * ORIGIN).word;
        if word.len() % 2 == 1 {
            // Odd elements swap the two triangle shades; one more mirror fixes that.
            word.push(0);
        }
        for k in word {
            self.m *= g.reflect(tiling.mirrors[k]);
            self.color_frame = tiling.quotient.reflect_right(self.color_frame, k);
        }
        self.m = g.orthonormalize(self.m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        geom::tests::{ALL, MU, MUS, assert_isometry, in_picture},
        quotient::E_C,
    };
    use std::f64::consts::PI;

    /// One tiling per geometry.
    const PAIRS: [(u32, u32); 3] = [(5, 3), (4, 4), (7, 3)];

    #[test]
    fn grabbed_point_stays_under_cursor() {
        for g in ALL {
            for mu in MUS {
                let mut v = View::new(g);
                let path = [
                    DVec2::new(0.1, 0.2),
                    DVec2::new(-0.3, 0.5),
                    DVec2::new(0.6, -0.4),
                ]
                .map(|d| in_picture(g, mu, d));
                let grabbed = v.m.inverse() * g.disk_to_point(path[0], mu);
                for w in path.windows(2) {
                    v.pan(w[0], w[1], mu);
                }
                let d = g.point_to_disk(v.m * grabbed, mu);
                assert!((d - path[2]).length() < 1e-9);
            }
        }
    }

    /// Folded tiling point, parity and tile color under screen point `d`, as
    /// the shader computes them.
    fn sample(v: &View, t: &Tiling, d: DVec2) -> (glam::DVec3, usize, [u32; 3]) {
        let f = t.fold(v.m.inverse() * t.geometry.disk_to_point(d, MU));
        let quot = &t.quotient;
        let acc = f
            .word
            .iter()
            .fold(v.color_frame, |a, &k| quot.reflect_right(a, k));
        (f.point, f.word.len() % 2, quot.apply(acc, E_C))
    }

    #[test]
    fn settle_preserves_picture_and_bounds_m() {
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let g = t.geometry;
            let (mut raw, mut settled) = (View::new(g), View::new(g));
            let probes = [
                DVec2::new(0.1, 0.0),
                DVec2::new(-0.5, 0.3),
                DVec2::new(0.2, -0.8),
            ];
            // A long wiggly pan; unsettled, a hyperbolic M's entries would
            // grow exponentially, a Euclidean one's linearly.
            for i in 0..3000 {
                let a = 0.3 * (i as f64 * 0.01).sin();
                let (from, to) = (DVec2::ZERO, DVec2::from_angle(a) * 0.05);
                settled.pan(from, to, MU);
                settled.rotate(DVec2::X * 0.5, DVec2::from_angle(0.01) * 0.5);
                settled.settle(&t);
                if i < 50 {
                    raw.pan(from, to, MU);
                    raw.rotate(DVec2::X * 0.5, DVec2::from_angle(0.01) * 0.5);
                }
                if i == 49 {
                    for d in probes {
                        let ((yr, pr, cr), (ys, ps, cs)) =
                            (sample(&raw, &t, d), sample(&settled, &t, d));
                        assert_eq!((pr, cr), (ps, cs), "{{{p},{q}}}");
                        assert!((yr - ys).abs().max_element() < 1e-6, "{yr} vs {ys}");
                    }
                    if g == Geometry::Hyperbolic {
                        assert!(raw.m.abs().to_cols_array().iter().any(|&x| x > 20.0));
                    }
                }
                assert!(settled.m.abs().to_cols_array().iter().all(|&x| x < 10.0));
            }
            assert_isometry(g, settled.m, 1e-12);
        }
    }

    /// All 2p triangles of a tile share its color; its neighbors across an edge
    /// generally do not.
    #[test]
    fn color_is_constant_on_tiles() {
        for (p, q) in PAIRS {
            let t = Tiling::new(p, q).unwrap();
            let g = t.geometry;
            let v = View::new(g);
            let at = |i: u32| DVec2::from_angle((i as f64 + 0.5) * PI / p as f64) * 0.1;
            let colors: Vec<_> = (0..2 * p).map(|i| sample(&v, &t, at(i)).2).collect();
            assert!(colors.iter().all(|&c| c == E_C));
            let across_edge =
                g.point_to_disk(g.reflect(t.mirrors[2]) * g.disk_to_point(at(0), MU), MU);
            assert_ne!(sample(&v, &t, across_edge).2, E_C);
        }
    }

    /// The same for apeirogons, whose triangles are the images of the
    /// fundamental one under the infinite dihedral group ⟨a,b⟩.
    #[test]
    fn color_is_constant_on_apeirogons() {
        use crate::tiling::Count::Infinite as INF;
        for q in [3.into(), 8.into(), INF] {
            let t = Tiling::new(INF, q).unwrap();
            let g = t.geometry;
            let v = View::new(g);
            let [a, b, c] = t.mirrors.map(|m| g.reflect(m));
            let x = g.disk_to_point(DVec2::new(0.02, 0.03), MU);
            assert!(t.fold(x).word.is_empty());
            let mut y = x;
            for _ in 0..6 {
                for z in [y, a * y] {
                    let (_, _, color) = sample(&v, &t, g.point_to_disk(z, MU));
                    assert_eq!(color, E_C);
                }
                y = a * b * y;
            }
            assert_ne!(sample(&v, &t, g.point_to_disk(c * x, MU)).2, E_C);
        }
    }

    #[test]
    fn rotate_follows_pointer_angle() {
        for g in ALL {
            let mut v = View::new(g);
            v.rotate(DVec2::new(0.5, 0.0), DVec2::new(0.0, 0.3));
            let d = g.point_to_disk(v.m * g.disk_to_point(DVec2::new(0.4, 0.0), MU), MU);
            assert!((d - DVec2::new(0.0, 0.4)).length() < 1e-12);
        }
    }
}
