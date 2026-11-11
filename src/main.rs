mod camera;
mod geom;
mod gpu;
mod quotient;
mod snapshot;
mod tiling;
mod ui;
mod view;

use std::{sync::Arc, time::Duration};

use glam::DVec2;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};

use camera::Camera;
use geom::{Chart, Projection};
use gpu::{PaneRect, Renderer, Uniforms};
use snapshot::Snapshot;
use tiling::Tiling;
use ui::{Action, Gui, Settings};
use view::View;
// std's Instant panics in the browser.
use web_time::Instant;

/// Gap between the edge of the picture and the window edge, px.
const MARGIN_PX: f32 = 8.0;
/// The 3D view's orbit per screen unit, half the pane's shorter side.
const ORBIT_PER_UNIT: f64 = std::f64::consts::FRAC_PI_2;
/// The factor by which one wheel notch, or 100 px of scrolling, dollies or
/// zooms the 3D view.
const WHEEL_NOTCH: f64 = 0.85;
/// Two presses at most this far apart in time and px make a double click.
const DOUBLE_CLICK: (Duration, f64) = (Duration::from_millis(400), 4.0);

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    /// Left drag: pan the chart, or orbit the 3D view's camera.
    Primary,
    /// Right or Shift drag: rotate the disk, scale the half-plane about ∞,
    /// slide the band along its axis, or move the camera's target.
    Secondary,
}

impl Drag {
    /// Whether the drag keeps the grabbed point under the cursor, so needs a
    /// point it can hold, see `View::pannable`.
    fn holds(self, chart: Chart) -> bool {
        self == Self::Primary || chart == Chart::HalfPlane
    }
}

/// The window and what draws into it.
enum Gfx {
    /// Before `resumed`.
    None,
    /// Waiting for the renderer, which arrives as a user event.
    Pending(Arc<Window>),
    Ready(Box<Ready>),
}

struct Ready {
    window: Arc<Window>,
    renderer: Renderer,
    gui: Gui,
}

/// A part of the window, and what it shows: a chart, or with none the 3D
/// view.
#[derive(Clone, Copy, PartialEq)]
struct Pane {
    rect: PaneRect,
    chart: Option<Chart>,
}

impl Pane {
    /// (centre x, centre y, scale) in window px, the scale per screen unit:
    /// the disk's radius, or the distance from the center to the half-plane's
    /// boundary and to the band's edges; for the 3D view, half the shorter
    /// side.
    fn geometry(&self) -> [f32; 3] {
        let [x, y, w, h] = self.rect.map(|v| v as f32);
        let fit = match self.chart {
            Some(Chart::Disk) => w.min(h),
            Some(Chart::HalfPlane | Chart::Band) => h,
            None => w.min(h) + 2.0 * MARGIN_PX,
        };
        [x + 0.5 * w, y + 0.5 * h, (0.5 * fit - MARGIN_PX).max(1.0)]
    }

    fn contains(&self, pos: PhysicalPosition<f64>) -> bool {
        let [x, y, w, h] = self.rect.map(f64::from);
        (x..x + w).contains(&pos.x) && (y..y + h).contains(&pos.y)
    }

    /// Window pixel position to screen coordinates: from the pane's center,
    /// y up as in the shader, in units of the scale.
    fn to_screen(self, pos: PhysicalPosition<f64>) -> DVec2 {
        let [cx, cy, scale] = self.geometry().map(f64::from);
        DVec2::new(pos.x - cx, cy - pos.y) / scale
    }
}

/// The panes of a window of `size`, in the order drawn: the main one, which
/// fills it, then in 3D the inset, if shown, at the bottom right. The 2D
/// pane shows `chart`.
fn panes([w, h]: [u32; 2], settings: &Settings, chart: Chart) -> Vec<Pane> {
    let full = [0, 0, w, h];
    if !settings.view_3d {
        return vec![Pane {
            rect: full,
            chart: Some(chart),
        }];
    }
    let (main, inset) = if settings.swapped {
        (Some(chart), None)
    } else {
        (None, Some(chart))
    };
    let mut panes = vec![Pane {
        rect: full,
        chart: main,
    }];
    if settings.inset {
        let side = (w.min(h) / 3).max(1);
        let gap = MARGIN_PX as u32;
        panes.push(Pane {
            rect: [
                w.saturating_sub(side + gap),
                h.saturating_sub(side + gap),
                side,
                side,
            ],
            chart: inset,
        });
    }
    panes
}

impl Ready {
    fn panes(&self, settings: &Settings, chart: Chart) -> Vec<Pane> {
        let (w, h) = self.renderer.size();
        panes([w, h], settings, chart)
    }
}

/// What is shown, and the input state that changes it.
struct Model {
    settings: Settings,
    tiling: Tiling,
    /// Number of quotient colors, if small enough to count.
    colors: Option<usize>,
    view: View,
    camera: Camera,
    cursor: Option<PhysicalPosition<f64>>,
    modifiers: ModifiersState,
    drag: Option<(MouseButton, Drag, Pane)>,
    /// The last left press, to detect a double click.
    last_press: Option<(Instant, PhysicalPosition<f64>)>,
    /// When egui next wants a frame.
    repaint_at: Option<Instant>,
}

impl Model {
    fn new(snapshot: Option<&Snapshot>) -> Self {
        let tiling = snapshot.map_or_else(|| Tiling::new(7, 3).unwrap(), |s| s.tiling);
        Self {
            settings: snapshot.map_or_else(Settings::default, |s| s.settings(Settings::default())),
            tiling,
            colors: tiling.quotient.orbit_size(),
            // As printed, without settling, so that it stays exactly the same.
            view: snapshot.map_or_else(|| View::new(tiling.geometry), |s| s.view),
            camera: snapshot.map_or_else(
                || Camera::new(tiling.geometry, Settings::default().extent),
                |s| s.camera,
            ),
            cursor: None,
            modifiers: ModifiersState::empty(),
            drag: None,
            last_press: None,
            repaint_at: None,
        }
    }

    fn projection(&self) -> Projection {
        self.settings.projection(self.tiling.geometry)
    }

    /// The pane at window position `pos`: the inset if it is there.
    fn pane_at(&self, pos: PhysicalPosition<f64>, gfx: &Ready) -> Pane {
        let panes = gfx.panes(&self.settings, self.projection().chart);
        *panes
            .iter()
            .rev()
            .find(|p| p.contains(pos))
            .unwrap_or(&panes[0])
    }

    fn reset_camera(&mut self) {
        self.camera = Camera::new(self.tiling.geometry, self.settings.extent);
    }

    fn title(&self) -> String {
        let (t, ell) = (&self.tiling, self.tiling.quotient.ell);
        let colors = match self.colors {
            Some(n) => format!("{n} colors mod {ell}"),
            None => format!("colors mod {ell}"),
        };
        format!("{{{},{}}} {} tiling, {colors}", t.p, t.q, t.geometry.name())
    }

    fn set_tiling(&mut self, tiling: Tiling, window: &Window) {
        // A view of one geometry means nothing in another.
        if tiling.geometry != self.tiling.geometry {
            self.view = View::new(tiling.geometry);
            self.camera = Camera::new(tiling.geometry, self.settings.extent);
        }
        self.tiling = tiling;
        self.colors = tiling.quotient.orbit_size();
        self.view.reset_colors();
        self.view.settle(&self.tiling);
        window.set_title(&self.title());
    }

    fn print_view(&self, window: &Window) {
        let size = window.inner_size();
        let snapshot = Snapshot::new(
            &self.tiling,
            &self.view,
            &self.camera,
            &self.settings,
            [size.width, size.height],
        );
        emit(&snapshot.to_string());
    }

    fn paint(&mut self, gfx: &mut Ready) {
        let debug = (self.settings.debug).then(|| snapshot::describe(&self.tiling, &self.view));
        let panes = gfx.panes(&self.settings, self.projection().chart);
        let inset = panes.get(1).map(|p| p.rect);
        let out = gfx.gui.run(
            &gfx.window,
            &mut self.settings,
            &self.tiling,
            self.colors,
            inset,
            debug.as_deref(),
        );
        // The panel was laid out before any action; redraw to show it.
        let repaint_after = match out.action {
            Some(Action::SetTiling(t)) => {
                self.set_tiling(t, &gfx.window);
                Duration::ZERO
            }
            Some(Action::ResetTiling) => {
                self.view = View::new(self.tiling.geometry);
                Duration::ZERO
            }
            Some(Action::ResetCamera) => {
                self.reset_camera();
                Duration::ZERO
            }
            Some(Action::PrintView) => {
                self.print_view(&gfx.window);
                out.repaint_after
            }
            None => out.repaint_after,
        };
        self.repaint_at = Instant::now().checked_add(repaint_after);
        let proj = self.projection();
        let panes = gfx.panes(&self.settings, proj.chart);
        let drawn: Vec<_> = (panes.iter())
            .map(|pane| {
                let camera = pane.chart.is_none().then_some(&self.camera);
                let u = Uniforms::new(
                    &self.view,
                    &self.tiling,
                    pane.geometry(),
                    proj,
                    camera,
                    &self.settings,
                );
                (u, pane.rect)
            })
            .collect();
        gfx.renderer.render(&drawn, out.overlay);
    }

    fn key(&mut self, code: KeyCode, window: &Window) {
        let step = if self.modifiers.shift_key() { -1 } else { 1 };
        match code {
            KeyCode::KeyP | KeyCode::KeyQ => {
                let (dp, dq) = if code == KeyCode::KeyP {
                    (step, 0)
                } else {
                    (0, step)
                };
                let Some(t) = self.tiling.stepped(dp, dq) else {
                    return;
                };
                self.set_tiling(t, window);
            }
            KeyCode::KeyL => {
                let Some(t) = self.tiling.with_ell_stepped(step.into()) else {
                    return;
                };
                self.set_tiling(t, window);
            }
            KeyCode::KeyC => self.settings.quotient_colors ^= true,
            KeyCode::KeyH => self.settings.show_panel ^= true,
            KeyCode::KeyD => self.settings.debug ^= true,
            KeyCode::KeyV => self.print_view(window),
            KeyCode::KeyR if self.modifiers.shift_key() => self.reset_camera(),
            KeyCode::KeyR => self.view = View::new(self.tiling.geometry),
            _ => return,
        }
        window.request_redraw();
    }

    /// Starts the drag that `button` makes with the current modifiers in
    /// `pane`, or switches an ongoing one to it.
    fn grab(&mut self, button: MouseButton, pane: Pane) {
        let proj = self.projection();
        let drag = match button {
            MouseButton::Left if self.modifiers.shift_key() => Drag::Secondary,
            MouseButton::Left => Drag::Primary,
            MouseButton::Right => Drag::Secondary,
            _ => return,
        };
        let Some(chart) = pane.chart else {
            self.drag = Some((button, drag, pane));
            return;
        };
        let held = self
            .cursor
            .is_some_and(|c| self.view.pannable(pane.to_screen(c), proj));
        self.drag = (held || !drag.holds(chart)).then_some((button, drag, pane));
    }

    /// Moves the chart's view or the camera for a drag in `pane` from `old`
    /// to `new`.
    fn drag(
        &mut self,
        drag: Drag,
        pane: Pane,
        old: PhysicalPosition<f64>,
        new: PhysicalPosition<f64>,
    ) {
        let [from, to] = [old, new].map(|p| pane.to_screen(p));
        let Some(chart) = pane.chart else {
            match drag {
                Drag::Primary => self.camera.orbit(-ORBIT_PER_UNIT * (to - from)),
                Drag::Secondary => self.camera.pan(from, to),
            }
            return;
        };
        let proj = self.projection();
        let before = self.view;
        match (drag, chart) {
            (Drag::Primary, _) => self.view.pan(from, to, proj),
            (Drag::Secondary, Chart::Disk) => self.view.rotate(from, to),
            (Drag::Secondary, Chart::HalfPlane) => self.view.scale(from, to, proj),
            (Drag::Secondary, Chart::Band) => self.view.slide(from, to),
        }
        // Past the edge the grabbed point can't follow; let go.
        if drag.holds(chart) && !self.view.pannable(to, proj) {
            self.drag = None;
        }
        self.view.settle(&self.tiling);
        if self.view.too_deep(&self.tiling) {
            self.view = before;
        }
    }

    /// In the 3D view, a double click makes the point of the surface under
    /// the pointer the camera's target, keeping its direction and distance.
    fn press(&mut self, pane: Pane) {
        let (Some(pos), None) = (self.cursor, pane.chart) else {
            return;
        };
        let now = Instant::now();
        let double = self.last_press.is_some_and(|(t, p)| {
            now.duration_since(t) <= DOUBLE_CLICK.0
                && DVec2::new(pos.x - p.x, pos.y - p.y).length() <= DOUBLE_CLICK.1
        });
        self.last_press = (!double).then_some((now, pos));
        if !double {
            return;
        }
        let g = self.tiling.geometry;
        let ray = self
            .camera
            .ray(pane.to_screen(pos), self.settings.fov.to_radians());
        if let Some(x) = camera::hit(g, camera::cap(g, self.settings.extent), &ray) {
            self.camera.target = x;
        }
    }

    /// A wheel or pinch of `notches`, positive away from the user or
    /// spreading, in `pane`: dollies the 3D view in, or if `zoom` zooms it
    /// in.
    fn wheel(&mut self, notches: f64, zoom: bool, pane: Pane) {
        if pane.chart.is_some() {
            return;
        }
        let factor = WHEEL_NOTCH.powf(notches);
        if zoom {
            self.camera.zoom(factor, &mut self.settings.fov);
        } else {
            self.camera.dolly(factor);
        }
    }

    fn window_event(&mut self, gfx: &mut Ready, event: WindowEvent) {
        // Not for egui, which would ask to repaint after every frame.
        if event == WindowEvent::RedrawRequested {
            self.paint(gfx);
            return;
        }
        let egui = gfx.gui.on_window_event(&gfx.window, &event);
        if egui.repaint {
            gfx.window.request_redraw();
        }
        match event {
            WindowEvent::Resized(size) => {
                gfx.renderer.resize(size.width, size.height);
                gfx.window.request_redraw();
            }
            WindowEvent::ModifiersChanged(m) => {
                self.modifiers = m.state();
                if let Some((button, _, pane)) = self.drag {
                    self.grab(button, pane);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if !egui.consumed => self.key(code, &gfx.window),
            WindowEvent::MouseInput {
                state: ElementState::Released,
                ..
            } => self.drag = None,
            WindowEvent::MouseInput { button, .. } if !egui.consumed => {
                let Some(pos) = self.cursor else {
                    return;
                };
                let pane = self.pane_at(pos, gfx);
                if button == MouseButton::Left {
                    self.press(pane);
                }
                self.grab(button, pane);
                gfx.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } if !egui.consumed => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y.into(),
                    MouseScrollDelta::PixelDelta(p) => p.y / 100.0,
                };
                if let Some(pos) = self.cursor {
                    let pane = self.pane_at(pos, gfx);
                    self.wheel(notches, self.modifiers.control_key(), pane);
                    gfx.window.request_redraw();
                }
            }
            // Browsers send pinches as Ctrl+wheel instead.
            WindowEvent::PinchGesture { delta, .. } if !egui.consumed => {
                if let Some(pos) = self.cursor {
                    let pane = self.pane_at(pos, gfx);
                    self.wheel(delta / (1.0 - WHEEL_NOTCH), true, pane);
                    gfx.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let (Some((_, drag, pane)), Some(old)) = (self.drag, self.cursor) {
                    self.drag(drag, pane, old, position);
                    gfx.window.request_redraw();
                }
                self.cursor = Some(position);
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            _ => {}
        }
    }
}

struct App {
    proxy: EventLoopProxy<Renderer>,
    gfx: Gfx,
    model: Model,
    /// The window size of a loaded view, for the native window.
    size: Option<[u32; 2]>,
}

impl ApplicationHandler<Renderer> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self.gfx, Gfx::None) {
            return;
        }
        let mut attrs = Window::default_attributes().with_title(self.model.title());
        if let Some([w, h]) = self.size {
            attrs = attrs.with_inner_size(winit::dpi::PhysicalSize::new(w, h));
        }
        #[cfg(target_arch = "wasm32")]
        let attrs = winit::platform::web::WindowAttributesExtWebSys::with_append(attrs, true);
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let renderer = Renderer::new(window.clone(), event_loop.owned_display_handle());
        let proxy = self.proxy.clone();
        let send = async move {
            // Fails only if the event loop has already exited.
            let _ = proxy.send_event(renderer.await);
        };
        // The browser can't block, so there the renderer arrives later.
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(send);
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(send);
        self.gfx = Gfx::Pending(window);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, mut renderer: Renderer) {
        let Gfx::Pending(window) = &self.gfx else {
            return;
        };
        let window = window.clone();
        // Resizes while pending were dropped; catch up.
        let size = window.inner_size();
        renderer.resize(size.width, size.height);
        let gui = Gui::new(&window, event_loop, renderer.max_texture_side());
        window.request_redraw();
        self.gfx = Gfx::Ready(Box::new(Ready {
            window,
            renderer,
            gui,
        }));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if event == WindowEvent::CloseRequested {
            event_loop.exit();
        } else if let Gfx::Ready(gfx) = &mut self.gfx {
            self.model.window_event(gfx, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Gfx::Ready(gfx) = &self.gfx else {
            return;
        };
        match self.model.repaint_at {
            Some(t) if t <= Instant::now() => {
                self.model.repaint_at = None;
                gfx.window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Some(t) => event_loop.set_control_flow(ControlFlow::WaitUntil(t)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }
}

/// The value after the command line option `name`.
#[cfg(not(target_arch = "wasm32"))]
fn arg(name: &str) -> Option<String> {
    let mut args = std::env::args().skip_while(|a| a != name);
    args.next()?;
    args.next()
}

/// The line given with `--view` (native) or as a `#view=` URL fragment (web).
fn view_line() -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    return arg("--view");
    #[cfg(target_arch = "wasm32")]
    {
        let hash = web_sys::window()?.location().hash().ok()?;
        hash.strip_prefix("#view=").map(str::to_owned)
    }
}

/// Writes the picture of `snapshot`, without the panel, to a PNG file.
#[cfg(not(target_arch = "wasm32"))]
fn export(snapshot: &Snapshot, path: &str) -> Result<(), png::EncodingError> {
    let settings = snapshot.settings(Settings::default());
    let proj = settings.projection(snapshot.tiling.geometry);
    let main = panes(snapshot.size, &settings, proj.chart)[0];
    let u = Uniforms::new(
        &snapshot.view,
        &snapshot.tiling,
        main.geometry(),
        proj,
        main.chart.is_none().then_some(&snapshot.camera),
        &settings,
    );
    let pixels = pollster::block_on(gpu::render_image(&u, snapshot.size));
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let [width, height] = snapshot.size;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    encoder.write_header()?.write_image_data(&pixels)
}

/// Prints a line to stdout, or in the browser to the console, there with
/// the page's address for the view as well.
fn emit(line: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    println!("{line}");
    #[cfg(target_arch = "wasm32")]
    {
        let href = web_sys::window().and_then(|w| w.location().href().ok());
        let page = href
            .as_deref()
            .map_or("", |h| h.split('#').next().unwrap_or(h));
        let text = format!("{line}\n{page}#view={line}");
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&text));
    }
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    let snapshot = view_line().map(|line| Snapshot::parse(&line));
    let snapshot = match snapshot {
        Some(Err(e)) => {
            let message = format!("bad view line: {e}");
            #[cfg(not(target_arch = "wasm32"))]
            {
                eprintln!("{message}");
                std::process::exit(2);
            }
            #[cfg(target_arch = "wasm32")]
            {
                web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(&message));
                None
            }
        }
        s => s.map(Result::unwrap),
    };
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = arg("--export") {
        let Some(snapshot) = &snapshot else {
            eprintln!("--export needs --view");
            std::process::exit(2);
        };
        if let Err(e) = export(snapshot, &path) {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
        return;
    }
    let event_loop = EventLoop::with_user_event()
        .build()
        .expect("create event loop");
    let app = App {
        proxy: event_loop.create_proxy(),
        gfx: Gfx::None,
        model: Model::new(snapshot.as_ref()),
        // The browser sizes the canvas itself.
        size: snapshot
            .as_ref()
            .map(|s| s.size)
            .filter(|_| cfg!(not(target_arch = "wasm32"))),
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).expect("run event loop");
    }
    // Returns at once; the browser drives the loop.
    #[cfg(target_arch = "wasm32")]
    winit::platform::web::EventLoopExtWebSys::spawn_app(event_loop, app);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(swapped: bool, inset: bool) -> Settings {
        Settings {
            view_3d: true,
            swapped,
            inset,
            ..Settings::default()
        }
    }

    #[test]
    fn the_inset_sits_inside_the_bottom_right_corner() {
        let size = [900, 600];
        let [main, inset] = panes(size, &settings(false, true), Chart::Band)[..] else {
            panic!("two panes");
        };
        assert_eq!(main.rect, [0, 0, 900, 600]);
        assert_eq!((main.chart, inset.chart), (None, Some(Chart::Band)));
        let [x, y, w, h] = inset.rect;
        assert_eq!((w, h), (200, 200));
        assert_eq!([x + w, y + h], [900 - 8, 600 - 8]);
        // The 3D view's scale is half its shorter side.
        assert_eq!(main.geometry(), [450.0, 300.0, 300.0]);
        let [cx, cy, _] = inset.geometry();
        assert_eq!([cx, cy], [x as f32 + 100.0, y as f32 + 100.0]);
        let centre = PhysicalPosition::new(cx as f64, cy as f64);
        assert!(inset.contains(centre) && inset.to_screen(centre) == DVec2::ZERO);
    }

    #[test]
    fn swapping_and_hiding_change_the_panes() {
        let size = [900, 600];
        let swapped = panes(size, &settings(true, true), Chart::Disk);
        assert_eq!(
            (swapped[0].chart, swapped[1].chart),
            (Some(Chart::Disk), None)
        );
        assert_eq!(panes(size, &settings(false, false), Chart::Disk).len(), 1);
        let flat = panes(size, &Settings::default(), Chart::Disk);
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].chart, Some(Chart::Disk));
    }
}
