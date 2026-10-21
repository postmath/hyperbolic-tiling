//! The egui control panel, drawn over the tiling.

use std::time::Duration;

use egui::{
    Align2, Color32, CursorIcon, FontId, Pos2, Rect, Sense, Slider, Stroke, Ui, Vec2, vec2,
};
use winit::{event::WindowEvent, raw_window_handle::HasDisplayHandle, window::Window};

use crate::{
    gpu::Overlay,
    tiling::{Geometry, MAX_PQ, Tiling},
};

pub const USAGE: &str = "\
Drag: pan. Right drag or Shift+drag: rotate. R: reset view.
P / Shift+P: step p up / down. Q / Shift+Q: step q.
L / Shift+L: step the prime ℓ for tile colors. C: toggle tile colors / two-tone.
H: hide / show the control panel.";

const LATTICE_PX: f32 = 280.0;

pub struct Settings {
    /// Color tiles by the finite quotient, rather than two-tone by parity.
    pub quotient_colors: bool,
    pub line_px: f32,
    /// Vertex dot radius.
    pub vertex_px: f32,
    pub show_panel: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            quotient_colors: true,
            line_px: 1.5,
            vertex_px: 3.5,
            show_panel: true,
        }
    }
}

pub enum Action {
    SetTiling(Tiling),
    ResetView,
}

pub struct Output {
    pub overlay: Overlay,
    pub action: Option<Action>,
    /// Delay until egui wants another frame; `Duration::MAX` for none.
    pub repaint_after: Duration,
}

pub struct Gui {
    ctx: egui::Context,
    state: egui_winit::State,
}

impl Gui {
    pub fn new(window: &Window, display: &dyn HasDisplayHandle, max_texture_side: usize) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            display,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(max_texture_side),
        );
        Self { ctx, state }
    }

    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    /// Lays out the panel for one frame. Settings change in place; changes to
    /// the tiling or view are returned for the caller to apply.
    pub fn run(
        &mut self,
        window: &Window,
        settings: &mut Settings,
        tiling: &Tiling,
        colors: Option<usize>,
    ) -> Output {
        let input = self.state.take_egui_input(window);
        let mut action = None;
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = self.ctx.run_pass(input, |ctx| {
            if settings.show_panel {
                egui::Window::new("Controls")
                    .anchor(Align2::LEFT_TOP, [8.0, 8.0])
                    .resizable(false)
                    .show(ctx, |ui| {
                        // A discarded pass is redone without input; keep its action.
                        action = panel(ui, settings, tiling, colors).or(action.take());
                    });
            }
        });
        self.state.handle_platform_output(window, platform_output);
        Output {
            overlay: Overlay {
                primitives: self.ctx.tessellate(shapes, pixels_per_point),
                textures: textures_delta,
                pixels_per_point,
            },
            action,
            repaint_after: viewport_output
                .get(&egui::ViewportId::ROOT)
                .map_or(Duration::MAX, |v| v.repaint_delay),
        }
    }
}

fn panel(
    ui: &mut Ui,
    settings: &mut Settings,
    tiling: &Tiling,
    colors: Option<usize>,
) -> Option<Action> {
    ui.set_width(LATTICE_PX);
    let mut action = None;

    ui.label(format!("Tiling {{{},{}}}", tiling.p, tiling.q));
    if let Some((p, q)) = lattice(ui, tiling) {
        action = Tiling::new(p, q).map(Action::SetTiling);
    }
    legend(ui);
    ui.separator();

    ui.checkbox(&mut settings.quotient_colors, "Tile colors mod ℓ");
    ui.horizontal(|ui| {
        let down = tiling.with_ell_stepped(-1);
        if ui
            .add_enabled(down.is_some(), egui::Button::new("-"))
            .clicked()
        {
            action = down.map(Action::SetTiling);
        }
        ui.label(format!("ℓ = {}", tiling.quotient.ell));
        let up = tiling.with_ell_stepped(1);
        if ui
            .add_enabled(up.is_some(), egui::Button::new("+"))
            .clicked()
        {
            action = up.map(Action::SetTiling);
        }
        ui.label(match colors {
            Some(n) => format!("{n} colors"),
            None => "too many colors to count".into(),
        });
    });
    ui.separator();

    ui.add(Slider::new(&mut settings.line_px, 0.0..=5.0).text("Line width"));
    ui.add(Slider::new(&mut settings.vertex_px, 0.0..=10.0).text("Vertex size"));
    if ui.button("Reset view").clicked() {
        action = Some(Action::ResetView);
    }
    ui.collapsing("Help", |ui| ui.label(USAGE));
    action
}

/// Okabe–Ito colors, which stay distinct under common color blindness.
fn geometry_color(g: Geometry) -> Color32 {
    match g {
        Geometry::Spherical => Color32::from_rgb(0xd5, 0x5e, 0x00),
        Geometry::Euclidean => Color32::from_rgb(0xf0, 0xe4, 0x42),
        Geometry::Hyperbolic => Color32::from_rgb(0x56, 0xb4, 0xe9),
    }
}

fn geometry_name(g: Geometry) -> &'static str {
    match g {
        Geometry::Spherical => "spherical",
        Geometry::Euclidean => "Euclidean",
        Geometry::Hyperbolic => "hyperbolic",
    }
}

fn legend(ui: &mut Ui) {
    ui.horizontal(|ui| {
        for g in [
            Geometry::Hyperbolic,
            Geometry::Euclidean,
            Geometry::Spherical,
        ] {
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
            ui.painter()
                .circle_filled(rect.center(), 4.0, geometry_color(g));
            ui.small(geometry_name(g));
        }
    });
    ui.small("p across, q up");
}

fn pairs() -> impl Iterator<Item = (u32, u32)> {
    (3..=MAX_PQ).flat_map(|p| (3..=MAX_PQ).map(move |q| (p, q)))
}

/// Positions in the {p,q} lattice: p across, q up, both on a log scale so that
/// small values, where the geometry changes, get the most room.
struct Lattice {
    /// Position of {3,3}.
    origin: Pos2,
    /// Points per unit of ln.
    scale: f32,
}

impl Lattice {
    fn new(plot: Rect) -> Self {
        Self {
            origin: plot.left_bottom(),
            scale: plot.width().min(plot.height()) / (MAX_PQ as f32 / 3.0).ln(),
        }
    }

    fn pos(&self, p: f32, q: f32) -> Pos2 {
        self.origin + self.scale * vec2((p / 3.0).ln(), -(q / 3.0).ln())
    }

    fn at(&self, (p, q): (u32, u32)) -> Pos2 {
        self.pos(p as f32, q as f32)
    }

    /// A fixed fraction of the local spacing, which is about scale / max(p, q).
    fn radius(&self, (p, q): (u32, u32)) -> f32 {
        (0.3 * self.scale / p.max(q) as f32).max(1.0)
    }

    fn nearest(&self, to: Pos2) -> (u32, u32) {
        pairs()
            .min_by(|&a, &b| {
                let d = |pq| self.at(pq).distance_sq(to);
                d(a).total_cmp(&d(b))
            })
            .unwrap()
    }
}

/// Draws the lattice with the current tiling marked; returns the hyperbolic
/// {p,q} clicked, if any.
fn lattice(ui: &mut Ui, tiling: &Tiling) -> Option<(u32, u32)> {
    let (response, painter) = ui.allocate_painter(Vec2::splat(LATTICE_PX), Sense::click());
    // Room for tick labels left and below, and for the largest dots.
    let rect = response.rect;
    let lat = Lattice::new(Rect::from_min_max(
        rect.min + vec2(22.0, 6.0),
        rect.max - vec2(6.0, 22.0),
    ));
    let weak = ui.visuals().weak_text_color();
    let strong = ui.visuals().strong_text_color();

    // The Euclidean curve (p-2)(q-2) = 4, from {3,6} to {6,3}.
    let curve = (0..=30)
        .map(|i| {
            let p = 3.0 + 0.1 * i as f32;
            lat.pos(p, 2.0 + 4.0 / (p - 2.0))
        })
        .collect();
    painter.line(curve, Stroke::new(1.0, weak));

    let font = FontId::proportional(10.0);
    for n in [3, 4, 5, 6, 8, 12, 16, 24, 32] {
        let below = Pos2::new(lat.at((n, 3)).x, lat.origin.y + 10.0);
        let left = Pos2::new(lat.origin.x - 10.0, lat.at((3, n)).y);
        painter.text(below, Align2::CENTER_TOP, n, font.clone(), weak);
        painter.text(left, Align2::RIGHT_CENTER, n, font.clone(), weak);
    }

    for pq in pairs() {
        let color = geometry_color(Geometry::of(pq.0, pq.1));
        painter.circle_filled(lat.at(pq), lat.radius(pq), color);
    }
    let ring = |pq, color| {
        painter.circle_stroke(lat.at(pq), lat.radius(pq) + 3.5, Stroke::new(1.5, color));
    };
    let current = (tiling.p, tiling.q);
    painter.circle_filled(lat.at(current), lat.radius(current) + 1.0, strong);
    ring(current, strong);

    let hovered = lat.nearest(response.hover_pos()?);
    let geometry = Geometry::of(hovered.0, hovered.1);
    if hovered != current {
        ring(hovered, weak);
    }
    if geometry == Geometry::Hyperbolic {
        ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
    }
    let clicked = response.clicked();
    response.on_hover_text_at_pointer(format!(
        "{{{},{}}} {}",
        hovered.0,
        hovered.1,
        geometry_name(geometry)
    ));
    (clicked && geometry == Geometry::Hyperbolic).then_some(hovered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lattice() -> Lattice {
        Lattice::new(Rect::from_min_max(
            Pos2::new(22.0, 6.0),
            Pos2::new(LATTICE_PX - 6.0, LATTICE_PX - 22.0),
        ))
    }

    #[test]
    fn lattice_dots_are_separate_and_hit_themselves() {
        let lat = lattice();
        for pq in pairs() {
            assert_eq!(lat.nearest(lat.at(pq)), pq);
            let (p, q) = pq;
            for next in [(p + 1, q), (p, q + 1)] {
                if next.0 <= MAX_PQ && next.1 <= MAX_PQ {
                    let gap = lat.at(pq).distance(lat.at(next));
                    assert!(lat.radius(pq) + lat.radius(next) < gap, "{pq:?} {next:?}");
                }
            }
        }
    }

    #[test]
    fn lattice_fills_its_plot() {
        let lat = lattice();
        assert_eq!(lat.at((3, 3)), Pos2::new(22.0, LATTICE_PX - 22.0));
        let top_right = lat.at((MAX_PQ, MAX_PQ));
        assert!((top_right.x - (LATTICE_PX - 6.0)).abs() < 1e-3);
        assert!((top_right.y - 6.0).abs() < 1e-3);
    }
}
