//! The egui control panel, drawn over the tiling.

use std::time::Duration;

use egui::{
    Align2, Color32, CursorIcon, FontId, Pos2, Rect, Sense, Slider, Stroke, Ui, Vec2, vec2,
};
use winit::{event::WindowEvent, raw_window_handle::HasDisplayHandle, window::Window};

use crate::{
    geom::Geometry,
    gpu::Overlay,
    tiling::{Count, MAX_LAMBDA, MAX_PQ, Tiling},
};

const USAGE: &str = "\
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
    /// The projection, see `Geometry::disk_to_point`.
    pub mu: f64,
    pub show_panel: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            quotient_colors: true,
            line_px: 1.5,
            vertex_px: 3.5,
            mu: 0.5,
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
        // The page behind the panel is dark whatever the system prefers.
        ctx.set_theme(egui::Theme::Dark);
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

    // Short enough for one line, so that the lattice below never moves.
    ui.label(format!(
        "Tiling {{{},{}}}, {}",
        tiling.p,
        tiling.q,
        tiling.geometry.name()
    ));
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

    projection(ui, &mut settings.mu, tiling.geometry);
    ui.separator();

    ui.add(Slider::new(&mut settings.line_px, 0.0..=5.0).text("Line width"));
    ui.add(Slider::new(&mut settings.vertex_px, 0.0..=10.0).text("Vertex size"));
    if ui.button("Reset view").clicked() {
        action = Some(Action::ResetView);
    }
    ui.collapsing("Help", |ui| ui.label(USAGE));
    action
}

/// Projection presets with their μ; see `Geometry::disk_to_point`.
fn presets(g: Geometry) -> [(&'static str, f64); 3] {
    match g {
        Geometry::Spherical => [
            ("Gnomonic", 0.0),
            ("Stereographic", 0.5),
            ("Orthographic", 1.0),
        ],
        _ => [("Klein", 0.0), ("Poincaré", 0.5), ("Gans", 1.0)],
    }
}

/// The slider runs over μ, which reaches λ = μ/(1-μ) = ∞; its number field
/// shows and takes λ. Smart aim lands on exactly μ = 1/2 within a pixel of it.
fn projection(ui: &mut Ui, mu: &mut f64, g: Geometry) {
    ui.add_enabled_ui(g != Geometry::Euclidean, |ui| {
        ui.horizontal(|ui| {
            for (name, preset) in presets(g) {
                ui.selectable_value(mu, preset, name);
            }
        });
        ui.add(
            Slider::new(mu, 0.0..=1.0)
                .text("Projection λ")
                .custom_formatter(|mu, _| lambda_text(mu))
                .custom_parser(parse_lambda),
        );
    })
    .response
    .on_disabled_hover_text("All projections of the plane agree.");
}

fn lambda_text(mu: f64) -> String {
    if mu >= 1.0 {
        return "∞".into();
    }
    let s = format!("{:.3}", mu / (1.0 - mu));
    s.trim_end_matches('0').trim_end_matches('.').into()
}

/// λ to μ.
fn parse_lambda(s: &str) -> Option<f64> {
    let s = s.trim();
    let lambda = if s == "∞" {
        f64::INFINITY
    } else {
        s.parse().ok()?
    };
    match lambda {
        l if l == f64::INFINITY => Some(1.0),
        l if l >= 0.0 => Some(l / (1.0 + l)),
        _ => None,
    }
}

/// Okabe–Ito colors, which stay distinct under common color blindness.
fn geometry_color(g: Geometry) -> Color32 {
    match g {
        Geometry::Spherical => Color32::from_rgb(0xd5, 0x5e, 0x00),
        Geometry::Euclidean => Color32::from_rgb(0xf0, 0xe4, 0x42),
        Geometry::Hyperbolic => Color32::from_rgb(0x56, 0xb4, 0xe9),
    }
}

/// For pairs with a hyperideal label, which are hyperbolic too.
const HYPERIDEAL_COLOR: Color32 = Color32::from_rgb(0xcc, 0x79, 0xa7);

fn pair_color((p, q): (Count, Count)) -> Color32 {
    let hyperideal = |c| matches!(c, Count::Hyperideal(_));
    if hyperideal(p) || hyperideal(q) {
        HYPERIDEAL_COLOR
    } else {
        geometry_color(Geometry::of(p, q))
    }
}

fn legend(ui: &mut Ui) {
    ui.horizontal(|ui| {
        let entries = [
            Geometry::Hyperbolic,
            Geometry::Euclidean,
            Geometry::Spherical,
        ]
        .map(|g| (geometry_color(g), g.name()));
        for (color, name) in entries
            .into_iter()
            .chain([(HYPERIDEAL_COLOR, "hyperideal")])
        {
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
            ui.painter().circle_filled(rect.center(), 4.0, color);
            ui.small(name);
        }
    });
    ui.small("p across, q up; beyond ∞, 2cosh δ");
}

fn pairs() -> impl Iterator<Item = (Count, Count)> {
    Count::all().flat_map(|p| Count::all().map(move |q| (p, q)))
}

/// How far ∞ is placed beyond MAX_PQ, and the first hyperideal label beyond
/// ∞, in units of ln.
const INF_GAP: f32 = 0.2;
/// The spacing of hyperideal labels, in units of ln.
const HYPERIDEAL_STEP: f32 = 0.1;

/// Position of `c` along an axis, in units of ln from 3.
fn coord(c: Count) -> f32 {
    let inf = (MAX_PQ as f32 / 3.0).ln() + INF_GAP;
    match c {
        Count::Finite(m) => (m as f32 / 3.0).ln(),
        Count::Infinite => inf,
        Count::Hyperideal(n) => inf + INF_GAP + (n - 3) as f32 * HYPERIDEAL_STEP,
    }
}

/// Positions in the {p,q} lattice: p across, q up, both on a log scale so that
/// small values, where the geometry changes, get the most room; ∞ comes a
/// little after MAX_PQ, and the hyperideal labels evenly spaced after that.
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
            scale: plot.width().min(plot.height()) / coord(Count::Hyperideal(MAX_LAMBDA)),
        }
    }

    fn pos(&self, p: f32, q: f32) -> Pos2 {
        self.origin + self.scale * vec2((p / 3.0).ln(), -(q / 3.0).ln())
    }

    fn at(&self, (p, q): (Count, Count)) -> Pos2 {
        self.origin + self.scale * vec2(coord(p), -coord(q))
    }

    /// A fixed fraction of the local spacing, which is about scale / max(p, q),
    /// with ∞ and hyperideal labels counting as MAX_PQ.
    fn radius(&self, (p, q): (Count, Count)) -> f32 {
        let m = |c| match c {
            Count::Finite(m) => m,
            Count::Infinite | Count::Hyperideal(_) => MAX_PQ,
        };
        (0.3 * self.scale / m(p).max(m(q)) as f32).max(1.0)
    }

    fn nearest(&self, to: Pos2) -> (Count, Count) {
        pairs()
            .min_by(|&a, &b| {
                let d = |pq| self.at(pq).distance_sq(to);
                d(a).total_cmp(&d(b))
            })
            .unwrap()
    }
}

/// Draws the lattice with the current tiling marked; returns the {p,q}
/// clicked, if any.
fn lattice(ui: &mut Ui, tiling: &Tiling) -> Option<(Count, Count)> {
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
    // Hyperideal labels are ticked by 2cosh δ, in their own color.
    let ticks = [3, 4, 5, 6, 8, 12, 16, 24, 32].map(|m| (Count::Finite(m), m.to_string(), weak));
    let hyperideal = [3, 6, 10].map(|n| (Count::Hyperideal(n), n.to_string(), HYPERIDEAL_COLOR));
    let infinite = (Count::Infinite, Count::Infinite.to_string(), weak);
    for (c, label, color) in ticks.into_iter().chain([infinite]).chain(hyperideal) {
        let three = Count::Finite(3);
        let below = Pos2::new(lat.at((c, three)).x, lat.origin.y + 10.0);
        let left = Pos2::new(lat.origin.x - 10.0, lat.at((three, c)).y);
        painter.text(below, Align2::CENTER_TOP, &label, font.clone(), color);
        painter.text(left, Align2::RIGHT_CENTER, &label, font.clone(), color);
    }

    for pq in pairs() {
        painter.circle_filled(lat.at(pq), lat.radius(pq), pair_color(pq));
    }
    let ring = |pq, color| {
        painter.circle_stroke(lat.at(pq), lat.radius(pq) + 3.5, Stroke::new(1.5, color));
    };
    let current = (tiling.p, tiling.q);
    painter.circle_filled(lat.at(current), lat.radius(current) + 1.0, strong);
    ring(current, strong);

    let hovered = lat.nearest(response.hover_pos()?);
    if hovered != current {
        ring(hovered, weak);
    }
    ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
    let clicked = response.clicked();
    response.on_hover_text_at_pointer(describe(hovered));
    clicked.then_some(hovered)
}

/// "{p,q} geometry", and 2cosh δ of hyperideal labels.
fn describe((p, q): (Count, Count)) -> String {
    let mut text = format!("{{{p},{q}}} {}", Geometry::of(p, q).name());
    for (name, c) in [("p", p), ("q", q)] {
        if let Count::Hyperideal(n) = c {
            text += &format!(", 2cosh δ = {n} for {name}");
        }
    }
    text
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
            let next = [p.stepped(1).map(|p| (p, q)), q.stepped(1).map(|q| (p, q))];
            for next in next.into_iter().flatten() {
                let gap = lat.at(pq).distance(lat.at(next));
                assert!(lat.radius(pq) + lat.radius(next) < gap, "{pq:?} {next:?}");
            }
        }
    }

    #[test]
    fn lattice_fills_its_plot() {
        let lat = lattice();
        assert_eq!(
            lat.at((3.into(), 3.into())),
            Pos2::new(22.0, LATTICE_PX - 22.0)
        );
        let top = Count::Hyperideal(MAX_LAMBDA);
        let top_right = lat.at((top, top));
        assert!((top_right.x - (LATTICE_PX - 6.0)).abs() < 1e-3);
        assert!((top_right.y - 6.0).abs() < 1e-3);
    }

    #[test]
    fn lambda_round_trips_through_text() {
        for (mu, text) in [(0.0, "0"), (0.5, "1"), (0.75, "3"), (1.0, "∞")] {
            assert_eq!(lambda_text(mu), text);
            assert_eq!(parse_lambda(text), Some(mu));
        }
        assert_eq!(parse_lambda("inf"), Some(1.0));
        assert_eq!(parse_lambda("-1"), None);
    }
}
