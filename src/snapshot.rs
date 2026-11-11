//! The exact state of a picture as one line of text, for reporting and
//! reproducing it: V prints it, and `--view <line>` (native) or a
//! `#view=<line>` URL fragment (web) loads it. The line uses only characters
//! that need no quoting in a shell or escaping in a URL, and prints every
//! float so that it parses back to the same bits.

use std::fmt;

use glam::{DMat3, DVec3};

use crate::{
    camera::Camera,
    geom::Chart,
    quotient::Quotient,
    tiling::{Count, Tiling},
    ui::Settings,
    view::View,
};

const VERSION: &str = "v1";

pub struct Snapshot {
    pub tiling: Tiling,
    pub view: View,
    /// The settings the picture depends on; the panel's visibility it does
    /// not, so that is left out.
    pub mu: f64,
    /// The chart chosen, which the picture shows only where it applies.
    pub chart: Chart,
    pub line_px: f32,
    pub vertex_px: f32,
    pub quotient_colors: bool,
    pub view_3d: bool,
    /// The 3D view's chart in the main pane, which an export shows.
    pub swapped: bool,
    pub show_projection: bool,
    pub camera: Camera,
    pub extent: f64,
    pub fov: f64,
    /// Window size in physical pixels; it sets the scale.
    pub size: [u32; 2],
}

impl Snapshot {
    pub fn new(
        tiling: &Tiling,
        view: &View,
        camera: &Camera,
        settings: &Settings,
        size: [u32; 2],
    ) -> Self {
        Self {
            tiling: *tiling,
            view: *view,
            mu: settings.mu,
            chart: settings.chart,
            line_px: settings.line_px,
            vertex_px: settings.vertex_px,
            quotient_colors: settings.quotient_colors,
            view_3d: settings.view_3d,
            swapped: settings.swapped,
            show_projection: settings.show_projection,
            camera: *camera,
            extent: settings.extent,
            fov: settings.fov,
            size,
        }
    }

    /// The settings with the snapshot's values, the rest from `base`.
    pub fn settings(&self, base: Settings) -> Settings {
        Settings {
            mu: self.mu,
            chart: self.chart,
            line_px: self.line_px,
            vertex_px: self.vertex_px,
            quotient_colors: self.quotient_colors,
            view_3d: self.view_3d,
            swapped: self.swapped,
            show_projection: self.show_projection,
            extent: self.extent,
            fov: self.fov,
            ..base
        }
    }

    pub fn parse(line: &str) -> Result<Self, String> {
        let mut fields = line.trim().split(',');
        if fields.next() != Some(VERSION) {
            return Err(format!("a view line starts with {VERSION}"));
        }
        let fields: Vec<(&str, &str)> = fields
            .map(|f| f.split_once('=').ok_or(format!("no = in {f:?}")))
            .collect::<Result<_, _>>()?;
        let get = |key: &str| {
            let found = fields.iter().find(|(k, _)| *k == key);
            found.map(|(_, v)| *v).ok_or(format!("no {key}="))
        };
        let number = |key: &str| -> Result<f64, String> {
            get(key)?.parse().map_err(|e| format!("{key}: {e}"))
        };
        let list = |key: &str, len: usize| -> Result<Vec<&str>, String> {
            let items: Vec<_> = get(key)?.split(':').collect();
            match items.len() {
                n if n == len => Ok(items),
                n => Err(format!("{key}: {len} entries, not {n}")),
            }
        };
        let (p, q) = (parse_count(get("p")?)?, parse_count(get("q")?)?);
        let ell: u32 = get("ell")?.parse().map_err(|e| format!("ell: {e}"))?;
        let quotient = Quotient::new(p, q, ell).ok_or(format!("ℓ = {ell} is not admissible"))?;
        let tiling = Tiling::new(p, q).ok_or(format!("no tiling {{{p},{q}}}"))?;
        let tiling = Tiling { quotient, ..tiling };
        let m: Vec<f64> = (list("m", 9)?.iter())
            .map(|x| x.parse().map_err(|e| format!("m: {e}")))
            .collect::<Result<_, String>>()?;
        let frame: Vec<u32> = (list("frame", 9)?.iter())
            .map(|x| x.parse().map_err(|e| format!("frame: {e}")))
            .collect::<Result<_, String>>()?;
        let (w, h) = get("size")?.split_once('x').ok_or("size: WxH")?;
        // Only lines of the 3D view have this and the camera.
        let extent = get("extent").map_or(Ok(Settings::default().extent), |_| number("extent"))?;
        let pixels = |x: &str| x.parse::<u32>().map_err(|e| format!("size: {e}"));
        let size = [pixels(w)?, pixels(h)?];
        Ok(Self {
            view: View {
                geometry: tiling.geometry,
                m: DMat3::from_cols_slice(&m),
                color_frame: std::array::from_fn(|c| std::array::from_fn(|r| frame[3 * c + r])),
            },
            tiling,
            mu: number("mu")?,
            // Lines printed before there were charts have none.
            chart: get("chart").map_or(Ok(Chart::Disk), parse_chart)?,
            line_px: number("line")? as f32,
            vertex_px: number("dot")? as f32,
            quotient_colors: get("colors")? == "1",
            view_3d: get("d3").is_ok_and(|v| v == "1"),
            swapped: get("swap").is_ok_and(|v| v == "1"),
            show_projection: get("proj").is_ok_and(|v| v == "1"),
            camera: match get("cam") {
                Err(_) => Camera::new(tiling.geometry, extent),
                Ok(_) => {
                    let c: Vec<f64> = (list("cam", 6)?.iter())
                        .map(|x| x.parse().map_err(|e| format!("cam: {e}")))
                        .collect::<Result<_, String>>()?;
                    Camera {
                        target: DVec3::new(c[0], c[1], c[2]),
                        azimuth: c[3],
                        elevation: c[4],
                        half_height: c[5],
                    }
                }
            },
            extent,
            fov: get("fov").map_or(Ok(Settings::default().fov), |_| number("fov"))?,
            size,
        })
    }
}

impl fmt::Display for Snapshot {
    /// Floats print in Rust's shortest form that parses back to the same
    /// value, and M's entries with an exponent, since they range widely.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let t = &self.tiling;
        let m = self.view.m.to_cols_array().map(|x| format!("{x:e}"));
        let frame: Vec<_> = (self.view.color_frame.as_flattened().iter())
            .map(|x| x.to_string())
            .collect();
        write!(
            f,
            "{VERSION},p={},q={},ell={},mu={},chart={},size={}x{},m={},frame={},line={},dot={},colors={}",
            count_text(t.p),
            count_text(t.q),
            t.quotient.ell,
            self.mu,
            chart_text(self.chart),
            self.size[0],
            self.size[1],
            m.join(":"),
            frame.join(":"),
            self.line_px,
            self.vertex_px,
            self.quotient_colors as u8,
        )?;
        if self.view_3d {
            let c = &self.camera;
            let (t, az, el, h) = (c.target, c.azimuth, c.elevation, c.half_height);
            write!(
                f,
                ",d3=1,cam={}:{}:{}:{az}:{el}:{h},extent={},fov={}",
                t.x, t.y, t.z, self.extent, self.fov
            )?;
            if self.swapped {
                write!(f, ",swap=1")?;
            }
            if self.show_projection {
                write!(f, ",proj=1")?;
            }
        }
        Ok(())
    }
}

fn chart_text(c: Chart) -> &'static str {
    match c {
        Chart::Disk => "disk",
        Chart::HalfPlane => "half",
        Chart::Band => "band",
    }
}

fn parse_chart(s: &str) -> Result<Chart, String> {
    [Chart::Disk, Chart::HalfPlane, Chart::Band]
        .into_iter()
        .find(|&c| chart_text(c) == s)
        .ok_or(format!("chart: {s:?}"))
}

/// p or q in plain ASCII: 7, inf, or h3 for 2cosh δ = 3.
fn count_text(c: Count) -> String {
    match c {
        Count::Finite(m) => m.to_string(),
        Count::Infinite => "inf".into(),
        Count::Hyperideal(n) => format!("h{n}"),
    }
}

fn parse_count(s: &str) -> Result<Count, String> {
    let number = |s: &str| s.parse::<u32>().map_err(|e| format!("{s}: {e}"));
    match s {
        "inf" => Ok(Count::Infinite),
        _ => match s.strip_prefix('h') {
            Some(n) => Ok(Count::Hyperideal(number(n)?)),
            None => Ok(Count::Finite(number(s)?)),
        },
    }
}

/// What the debug section of the panel shows: M rounded, and how close the
/// view centre is to the pan limits.
pub fn describe(tiling: &Tiling, view: &View) -> String {
    let m = view.m;
    let mut text = String::from("M, rows:\n");
    for r in 0..3 {
        let row = DVec3::new(m.x_axis[r], m.y_axis[r], m.z_axis[r]);
        text += &format!("  {:>10.4} {:>10.4} {:>10.4}\n", row.x, row.y, row.z);
    }
    let m_inv = m.inverse();
    let norm = m_inv.abs().to_cols_array().into_iter().fold(0.0, f64::max);
    // The monospace font lacks superscripts.
    text += &format!(
        "max |M^-1| = {norm:.3} (pans stop at {})",
        crate::view::MAX_VIEW_NORM
    );
    let center = m_inv * crate::geom::ORIGIN;
    for j in tiling.cusps() {
        let corner = ["", "O", "V"][j];
        let width = tiling.cusp_width(j, center);
        let limit = crate::view::MIN_CUSP_WIDTH;
        text += &format!("\ncusp width at {corner}: {width:.3e} (pans stop at {limit:e})");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::tests::DISK;
    use glam::DVec2;

    #[test]
    fn round_trips_exactly() {
        use Count::{Hyperideal as H, Infinite as INF};
        for (p, q) in [
            (7.into(), 3.into()),
            (INF, H(3)),
            (H(4), INF),
            (4.into(), 4.into()),
        ] {
            let t = Tiling::new(p, q).unwrap();
            let t = t.with_ell_stepped(1).unwrap();
            let mut v = View::new(t.geometry);
            v.pan(DVec2::new(0.3, 0.1), DVec2::new(-0.2, 0.4), DISK);
            v.settle(&t);
            let settings = Settings {
                mu: 0.3,
                chart: Chart::Band,
                line_px: 2.25,
                quotient_colors: false,
                ..Settings::default()
            };
            let camera = Camera::new(t.geometry, settings.extent);
            let line = Snapshot::new(&t, &v, &camera, &settings, [1900, 1000]).to_string();
            let back = Snapshot::parse(&line).unwrap();
            assert_eq!(back.to_string(), line);
            assert_eq!((back.tiling.p, back.tiling.q), (p, q));
            assert_eq!(back.tiling.quotient.ell, t.quotient.ell);
            assert_eq!(back.view.m.to_cols_array(), v.m.to_cols_array());
            assert_eq!(back.view.color_frame, v.color_frame);
            assert_eq!(back.size, [1900, 1000]);
            assert_eq!(back.chart, Chart::Band);
            let old = Snapshot::parse(&line.replace(",chart=band", "")).unwrap();
            assert_eq!(old.chart, Chart::Disk);
            assert!(Snapshot::parse(&line.replace("chart=band", "chart=x")).is_err());
            assert!(!back.view_3d);
            assert!(!line.contains(|c: char| " ;&#'\"".contains(c)), "{line}");
        }
    }

    #[test]
    fn round_trips_the_3d_view() {
        let t = Tiling::new(7, 3).unwrap();
        let settings = Settings {
            view_3d: true,
            swapped: true,
            show_projection: true,
            extent: 3.25,
            fov: 0.0,
            ..Settings::default()
        };
        let camera = Camera {
            target: DVec3::new(1.5, -0.25, 0.125),
            azimuth: 0.1,
            elevation: -0.7,
            half_height: 3.3,
        };
        let line = Snapshot::new(&t, &View::new(t.geometry), &camera, &settings, [800, 600]);
        let line = line.to_string();
        let back = Snapshot::parse(&line).unwrap();
        assert_eq!(back.to_string(), line);
        assert!(back.view_3d && back.swapped && back.show_projection);
        assert_eq!(back.camera, camera);
        assert_eq!((back.extent, back.fov), (3.25, 0.0));
    }

    #[test]
    fn rejects_bad_lines() {
        for line in ["", "v2,p=7", "v1,p=7,q=3", "v1,p=x,q=3,ell=43"] {
            assert!(Snapshot::parse(line).is_err(), "{line:?}");
        }
    }
}
