//! The view isometry and the drags that change it.

use glam::{DMat3, DVec2};

use crate::{
    hyp::{ORIGIN, disk_to_hyp, lorentz_inv, orthonormalize, reflect, rotate, translate},
    quotient::{self, ModMat},
    tiling::Tiling,
};

/// Pan drags hold points up to this rho^2, where the shader starts its rim fade.
const PAN_RHO2_MAX: f64 = 0.98;
/// Rotate drags ignore pointers this close to the center, where the angle is noise.
const ROTATE_RHO_MIN: f64 = 0.02;

/// Whether a pan drag can hold the point at disk position `d`.
pub fn pannable(d: DVec2) -> bool {
    d.length_squared() <= PAN_RHO2_MAX
}

pub struct View {
    /// Tiling frame -> screen frame.
    pub m: DMat3,
    /// φ(G) for the symmetry G that `settle` has absorbed into `m`: a point x
    /// of the current tiling frame is G x in the frame the colors refer to.
    pub color_frame: ModMat,
}

impl View {
    pub fn new() -> Self {
        Self {
            m: DMat3::IDENTITY,
            color_frame: quotient::IDENTITY,
        }
    }

    /// Makes the current tiling frame the one colors refer to; needed when the
    /// quotient changes.
    pub fn reset_colors(&mut self) {
        self.color_frame = quotient::IDENTITY;
    }

    /// Moves the picture so the point under disk position `from` ends up under
    /// `to`. Points beyond `pannable` are clamped to its edge.
    pub fn pan(&mut self, from: DVec2, to: DVec2) {
        let (p, q) = (disk_to_hyp(clamp(from)), disk_to_hyp(clamp(to)));
        self.m = translate(p, q) * self.m;
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
        let mut word = tiling.fold(lorentz_inv(self.m) * ORIGIN).word;
        if word.len() % 2 == 1 {
            // Odd elements swap the two triangle shades; one more mirror fixes that.
            word.push(0);
        }
        for k in word {
            self.m *= reflect(tiling.mirrors[k]);
            self.color_frame = tiling.quotient.reflect_right(self.color_frame, k);
        }
        self.m = orthonormalize(self.m);
    }
}

fn clamp(d: DVec2) -> DVec2 {
    let max = PAN_RHO2_MAX.sqrt();
    if d.length() > max {
        d.normalize() * max
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyp::{hyp_to_disk, lorentz_inv};
    use std::f64::consts::PI;

    #[test]
    fn grabbed_point_stays_under_cursor() {
        let mut v = View::new();
        let path = [
            DVec2::new(0.1, 0.2),
            DVec2::new(-0.3, 0.5),
            DVec2::new(0.6, -0.4),
        ];
        let grabbed = lorentz_inv(v.m) * disk_to_hyp(path[0]);
        for w in path.windows(2) {
            v.pan(w[0], w[1]);
        }
        let d = hyp_to_disk(v.m * grabbed);
        assert!((d - path[2]).length() < 1e-9);
    }

    /// Folded tiling point, parity and tile color under screen point `d`, as
    /// the shader computes them.
    fn sample(v: &View, t: &Tiling, d: DVec2) -> (glam::DVec3, usize, [u32; 3]) {
        let f = t.fold(lorentz_inv(v.m) * disk_to_hyp(d));
        let quot = &t.quotient;
        let acc = f
            .word
            .iter()
            .fold(v.color_frame, |a, &k| quot.reflect_right(a, k));
        (f.point, f.word.len() % 2, quot.apply(acc, quot.v0))
    }

    #[test]
    fn settle_preserves_picture_and_bounds_m() {
        let t = Tiling::new(7, 3).unwrap();
        let (mut raw, mut settled) = (View::new(), View::new());
        let probes = [
            DVec2::new(0.1, 0.0),
            DVec2::new(-0.5, 0.3),
            DVec2::new(0.2, -0.8),
        ];
        // A long wiggly pan; unsettled, M's entries would grow exponentially.
        for i in 0..3000 {
            let a = 0.3 * (i as f64 * 0.01).sin();
            let (from, to) = (DVec2::ZERO, DVec2::from_angle(a) * 0.05);
            settled.pan(from, to);
            settled.rotate(DVec2::X * 0.5, DVec2::from_angle(0.01) * 0.5);
            settled.settle(&t);
            if i < 50 {
                raw.pan(from, to);
                raw.rotate(DVec2::X * 0.5, DVec2::from_angle(0.01) * 0.5);
            }
            if i == 49 {
                for d in probes {
                    let ((yr, pr, cr), (ys, ps, cs)) =
                        (sample(&raw, &t, d), sample(&settled, &t, d));
                    assert_eq!((pr, cr), (ps, cs));
                    assert!((yr - ys).abs().max_element() < 1e-6, "{yr} vs {ys}");
                }
                assert!(raw.m.abs().to_cols_array().iter().any(|&x| x > 20.0));
            }
            assert!(settled.m.abs().to_cols_array().iter().all(|&x| x < 10.0));
        }
        crate::hyp::tests::assert_lorentz(settled.m, 1e-12);
    }

    /// All 2p triangles of a tile share its color; its neighbors across an edge
    /// generally do not.
    #[test]
    fn color_is_constant_on_tiles() {
        let t = Tiling::new(7, 3).unwrap();
        let v = View::new();
        let colors: Vec<_> = (0..14)
            .map(|i| sample(&v, &t, DVec2::from_angle((i as f64 + 0.5) * PI / 7.0) * 0.1).2)
            .collect();
        assert!(colors.iter().all(|&c| c == t.quotient.v0));
        let across_edge = sample(&v, &t, DVec2::from_angle(PI / 7.0) * 0.5).2;
        assert_ne!(across_edge, t.quotient.v0);
    }

    #[test]
    fn rotate_follows_pointer_angle() {
        let mut v = View::new();
        v.rotate(DVec2::new(0.5, 0.0), DVec2::new(0.0, 0.3));
        let d = hyp_to_disk(v.m * disk_to_hyp(DVec2::new(0.4, 0.0)));
        assert!((d - DVec2::new(0.0, 0.4)).length() < 1e-12);
    }
}
