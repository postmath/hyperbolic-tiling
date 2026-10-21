mod gpu;
mod hyp;
mod quotient;
mod tiling;
mod ui;
mod view;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use glam::DVec2;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};

use gpu::{Renderer, Uniforms};
use tiling::Tiling;
use ui::{Action, Gui, Settings};
use view::View;

/// Gap between the disk and the nearer window edge, px.
const MARGIN_PX: f32 = 8.0;

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Pan,
    Rotate,
}

/// The window and what draws into it.
enum Gfx {
    /// Before `resumed`.
    None,
    /// Waiting for the renderer, which arrives as a user event.
    Pending(Arc<Window>),
    Ready(Ready),
}

struct Ready {
    window: Arc<Window>,
    renderer: Renderer,
    gui: Gui,
}

impl Ready {
    /// (width, height, disk radius) in px.
    fn geometry(&self) -> (f32, f32, f32) {
        let (w, h) = self.renderer.size();
        let (w, h) = (w as f32, h as f32);
        (w, h, (0.5 * w.min(h) - MARGIN_PX).max(1.0))
    }

    /// Window pixel position to disk coordinates, y up as in the shader.
    fn to_disk(&self, pos: PhysicalPosition<f64>) -> DVec2 {
        let (w, h, r) = self.geometry();
        DVec2::new(pos.x - 0.5 * w as f64, 0.5 * h as f64 - pos.y) / r as f64
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
    drag: Option<Drag>,
    /// When egui next wants a frame.
    repaint_at: Option<Instant>,
}

impl Model {
    fn new() -> Self {
        let tiling = Tiling::new(7, 3).unwrap();
        Self {
            settings: Settings::default(),
            tiling,
            colors: tiling.quotient.orbit_size(),
            view: View::new(),
            cursor: None,
            modifiers: ModifiersState::empty(),
            drag: None,
            repaint_at: None,
        }
    }

    fn title(&self) -> String {
        let (t, ell) = (&self.tiling, self.tiling.quotient.ell);
        let colors = match self.colors {
            Some(n) => format!("{n} colors mod {ell}"),
            None => format!("colors mod {ell}"),
        };
        format!("{{{},{}}} hyperbolic tiling, {colors}", t.p, t.q)
    }

    fn set_tiling(&mut self, tiling: Tiling, window: &Window) {
        self.tiling = tiling;
        self.colors = tiling.quotient.orbit_size();
        self.view.reset_colors();
        self.view.settle(&self.tiling);
        window.set_title(&self.title());
    }

    fn paint(&mut self, gfx: &mut Ready) {
        let out = gfx
            .gui
            .run(&gfx.window, &mut self.settings, &self.tiling, self.colors);
        // The panel was laid out before any action; redraw to show it.
        let repaint_after = match out.action {
            Some(Action::SetTiling(t)) => {
                self.set_tiling(t, &gfx.window);
                Duration::ZERO
            }
            Some(Action::ResetView) => {
                self.view = View::new();
                Duration::ZERO
            }
            None => out.repaint_after,
        };
        self.repaint_at = Instant::now().checked_add(repaint_after);
        let (w, h, r) = gfx.geometry();
        let u = Uniforms::new(&self.view, &self.tiling, [w, h, r], &self.settings);
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
            KeyCode::KeyR => self.view = View::new(),
            _ => return,
        }
        window.request_redraw();
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
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
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
                self.drag = match button {
                    MouseButton::Left if self.modifiers.shift_key() => Some(Drag::Rotate),
                    MouseButton::Left
                        if self.cursor.is_some_and(|c| view::pannable(gfx.to_disk(c))) =>
                    {
                        Some(Drag::Pan)
                    }
                    MouseButton::Right => Some(Drag::Rotate),
                    _ => self.drag,
                };
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let (Some(drag), Some(old)) = (self.drag, self.cursor) {
                    let (from, to) = (gfx.to_disk(old), gfx.to_disk(position));
                    match drag {
                        Drag::Pan => {
                            self.view.pan(from, to);
                            // Past the rim the grabbed point can't follow; let go.
                            if !view::pannable(to) {
                                self.drag = None;
                            }
                        }
                        Drag::Rotate => self.view.rotate(from, to),
                    }
                    self.view.settle(&self.tiling);
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
}

impl ApplicationHandler<Renderer> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self.gfx, Gfx::None) {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title(self.model.title()))
                .expect("create window"),
        );
        let renderer = Renderer::new(window.clone(), event_loop.owned_display_handle());
        let proxy = self.proxy.clone();
        let send = async move {
            // Fails only if the event loop has already exited.
            let _ = proxy.send_event(renderer.await);
        };
        pollster::block_on(send);
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
        self.gfx = Gfx::Ready(Ready {
            window,
            renderer,
            gui,
        });
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

fn main() {
    let event_loop = EventLoop::with_user_event()
        .build()
        .expect("create event loop");
    let mut app = App {
        proxy: event_loop.create_proxy(),
        gfx: Gfx::None,
        model: Model::new(),
    };
    event_loop.run_app(&mut app).expect("run event loop");
}
