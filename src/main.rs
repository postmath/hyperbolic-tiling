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
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};

use geom::{Chart, Projection};
use gpu::{Renderer, Uniforms};
use snapshot::Snapshot;
use tiling::Tiling;
use ui::{Action, Gui, Settings};
use view::View;
// std's Instant panics in the browser.
use web_time::Instant;

/// Gap between the edge of the picture and the window edge, px.
const MARGIN_PX: f32 = 8.0;

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Pan,
    /// Right or Shift drag: rotate the disk, scale the half-plane about ∞, or
    /// slide the band along its axis.
    Secondary,
}

impl Drag {
    /// Whether the drag keeps the grabbed point under the cursor, so needs a
    /// point it can hold, see `View::pannable`.
    fn holds(self, chart: Chart) -> bool {
        self == Self::Pan || chart == Chart::HalfPlane
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

/// (width, height, scale) in px for a picture of `size`, the scale per
/// screen unit: the disk's radius, or the distance from the center to the
/// half-plane's boundary and to the band's edges.
fn geometry([w, h]: [u32; 2], chart: Chart) -> (f32, f32, f32) {
    let (w, h) = (w as f32, h as f32);
    let fit = match chart {
        Chart::Disk => w.min(h),
        Chart::HalfPlane | Chart::Band => h,
    };
    (w, h, (0.5 * fit - MARGIN_PX).max(1.0))
}

impl Ready {
    fn geometry(&self, chart: Chart) -> (f32, f32, f32) {
        let (w, h) = self.renderer.size();
        geometry([w, h], chart)
    }

    /// Window pixel position to screen coordinates: from the center, y up as
    /// in the shader, in units of the scale.
    fn to_screen(&self, pos: PhysicalPosition<f64>, chart: Chart) -> DVec2 {
        let (w, h, scale) = self.geometry(chart);
        DVec2::new(pos.x - 0.5 * w as f64, 0.5 * h as f64 - pos.y) / scale as f64
    }
}

/// What is shown, and the input state that changes it.
struct Model {
    settings: Settings,
    tiling: Tiling,
    /// Number of quotient colors, if small enough to count.
    colors: Option<usize>,
    view: View,
    cursor: Option<PhysicalPosition<f64>>,
    modifiers: ModifiersState,
    drag: Option<(MouseButton, Drag)>,
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
            cursor: None,
            modifiers: ModifiersState::empty(),
            drag: None,
            repaint_at: None,
        }
    }

    fn projection(&self) -> Projection {
        self.settings.projection(self.tiling.geometry)
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
            &self.settings,
            [size.width, size.height],
        );
        emit(&snapshot.to_string());
    }

    fn paint(&mut self, gfx: &mut Ready) {
        let debug = (self.settings.debug).then(|| snapshot::describe(&self.tiling, &self.view));
        let out = gfx.gui.run(
            &gfx.window,
            &mut self.settings,
            &self.tiling,
            self.colors,
            debug.as_deref(),
        );
        // The panel was laid out before any action; redraw to show it.
        let repaint_after = match out.action {
            Some(Action::SetTiling(t)) => {
                self.set_tiling(t, &gfx.window);
                Duration::ZERO
            }
            Some(Action::ResetView) => {
                self.view = View::new(self.tiling.geometry);
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
        let (w, h, scale) = gfx.geometry(proj.chart);
        let u = Uniforms::new(
            &self.view,
            &self.tiling,
            [w, h, scale],
            proj,
            &self.settings,
        );
        gfx.renderer.render(&u, out.overlay);
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
            KeyCode::KeyR => self.view = View::new(self.tiling.geometry),
            _ => return,
        }
        window.request_redraw();
    }

    /// Starts the drag that `button` makes with the current modifiers, or
    /// switches an ongoing one to it.
    fn grab(&mut self, button: MouseButton, gfx: &Ready) {
        let proj = self.projection();
        let drag = match button {
            MouseButton::Left if self.modifiers.shift_key() => Drag::Secondary,
            MouseButton::Left => Drag::Pan,
            MouseButton::Right => Drag::Secondary,
            _ => return,
        };
        let held = self
            .cursor
            .is_some_and(|c| self.view.pannable(gfx.to_screen(c, proj.chart), proj));
        self.drag = (held || !drag.holds(proj.chart)).then_some((button, drag));
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
                if let Some((button, _)) = self.drag {
                    self.grab(button, gfx);
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
            WindowEvent::MouseInput { button, .. } if !egui.consumed => self.grab(button, gfx),
            WindowEvent::CursorMoved { position, .. } => {
                if let (Some((_, drag)), Some(old)) = (self.drag, self.cursor) {
                    let proj = self.projection();
                    let [from, to] = [old, position].map(|p| gfx.to_screen(p, proj.chart));
                    let before = self.view;
                    match (drag, proj.chart) {
                        (Drag::Pan, _) => self.view.pan(from, to, proj),
                        (Drag::Secondary, Chart::Disk) => self.view.rotate(from, to),
                        (Drag::Secondary, Chart::HalfPlane) => self.view.scale(from, to, proj),
                        (Drag::Secondary, Chart::Band) => self.view.slide(from, to),
                    }
                    // Past the edge the grabbed point can't follow; let go.
                    if drag.holds(proj.chart) && !self.view.pannable(to, proj) {
                        self.drag = None;
                    }
                    self.view.settle(&self.tiling);
                    if self.view.too_deep(&self.tiling) {
                        self.view = before;
                    }
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
    let (w, h, scale) = geometry(snapshot.size, proj.chart);
    let u = Uniforms::new(
        &snapshot.view,
        &snapshot.tiling,
        [w, h, scale],
        proj,
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
