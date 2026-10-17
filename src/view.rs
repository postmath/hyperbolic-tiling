//! The view isometry and the drags that change it.

use glam::{DMat3, DVec2};

use crate::hyp::{disk_to_hyp, rotate, translate};

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

    #[test]
    fn rotate_follows_pointer_angle() {
        let mut v = View::new();
        v.rotate(DVec2::new(0.5, 0.0), DVec2::new(0.0, 0.3));
        let d = hyp_to_disk(v.m * disk_to_hyp(DVec2::new(0.4, 0.0)));
        assert!((d - DVec2::new(0.0, 0.4)).length() < 1e-12);
    }
}
