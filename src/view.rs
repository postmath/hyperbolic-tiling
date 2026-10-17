//! The view isometry and the drags that change it.

use glam::{DMat3, DVec2};

use crate::{
    hyp::{ORIGIN, disk_to_hyp, lorentz_inv, orthonormalize, reflect, rotate, translate},
    tiling::Tiling,
};

/// Pan pointers are clamped to this rho^2, where the shader starts its rim fade.
const PAN_RHO2_MAX: f64 = 0.98;
/// Rotate drags ignore pointers this close to the center, where the angle is noise.
const ROTATE_RHO_MIN: f64 = 0.02;

pub struct View {
    /// Tiling frame -> screen frame.
    pub m: DMat3,
}

impl View {
    pub fn new() -> Self {
        Self { m: DMat3::IDENTITY }
    }

    /// Moves the picture so the point under disk position `from` ends up under `to`.
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
    pub fn settle(&mut self, tiling: &Tiling) {
        let (_, mut f, count, _) = tiling.fold(lorentz_inv(self.m) * ORIGIN);
        if count % 2 == 1 {
            // Odd elements swap the two triangle colors; one more mirror fixes that.
            f = reflect(tiling.mirrors[0]) * f;
        }
        self.m = orthonormalize(self.m * lorentz_inv(f));
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

    /// Folded tiling point and parity under screen point `d`, as the shader sees it.
    fn sample(v: &View, t: &Tiling, d: DVec2) -> (glam::DVec3, u32) {
        let (y, _, count, _) = t.fold(lorentz_inv(v.m) * disk_to_hyp(d));
        (y, count % 2)
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
                    let ((yr, pr), (ys, ps)) = (sample(&raw, &t, d), sample(&settled, &t, d));
                    assert_eq!(pr, ps);
                    assert!((yr - ys).abs().max_element() < 1e-6, "{yr} vs {ys}");
                }
                assert!(raw.m.abs().to_cols_array().iter().any(|&x| x > 20.0));
            }
            assert!(settled.m.abs().to_cols_array().iter().all(|&x| x < 10.0));
        }
        crate::hyp::tests::assert_lorentz(settled.m, 1e-12);
    }

    #[test]
    fn rotate_follows_pointer_angle() {
        let mut v = View::new();
        v.rotate(DVec2::new(0.5, 0.0), DVec2::new(0.0, 0.3));
        let d = hyp_to_disk(v.m * disk_to_hyp(DVec2::new(0.4, 0.0)));
        assert!((d - DVec2::new(0.0, 0.4)).length() < 1e-12);
    }
}
