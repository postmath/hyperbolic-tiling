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
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
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

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    gui: Option<Gui>,
    /// When egui next wants a frame.
    repaint_at: Option<Instant>,
    settings: Settings,
    tiling: Tiling,
    view: View,
    cursor: Option<PhysicalPosition<f64>>,
    modifiers: ModifiersState,
    drag: Option<Drag>,
    /// Number of quotient colors, if small enough to count.
    colors: Option<usize>,
}

impl App {
    fn new() -> Self {
        let tiling = Tiling::new(7, 3).unwrap();
        Self {
            window: None,
            renderer: None,
            gui: None,
            repaint_at: None,
            settings: Settings::default(),
            tiling,
            colors: tiling.quotient.orbit_size(),
            view: View::new(),
            cursor: None,
            modifiers: ModifiersState::empty(),
            drag: None,
        }
    }

    /// (width, height, disk radius) in px.
    fn geometry(&self) -> (f32, f32, f32) {
        let (w, h) = self.renderer.as_ref().unwrap().size();
        let (w, h) = (w as f32, h as f32);
        (w, h, (0.5 * w.min(h) - MARGIN_PX).max(1.0))
    }

    /// Window pixel position to disk coordinates, y up as in the shader.
    fn to_disk(&self, pos: PhysicalPosition<f64>) -> DVec2 {
        let (w, h, r) = self.geometry();
        DVec2::new(pos.x - 0.5 * w as f64, 0.5 * h as f64 - pos.y) / r as f64
    }

    fn uniforms(&self) -> Uniforms {
        let (w, h, r) = self.geometry();
        Uniforms::new(&self.view, &self.tiling, [w, h, r], &self.settings)
    }

    fn redraw(&self) {
        self.window.as_ref().unwrap().request_redraw();
    }

    fn title(&self) -> String {
        let (t, ell) = (&self.tiling, self.tiling.quotient.ell);
        let colors = match self.colors {
            Some(n) => format!("{n} colors mod {ell}"),
            None => format!("colors mod {ell}"),
        };
        format!("{{{},{}}} hyperbolic tiling, {colors}", t.p, t.q)
    }

    fn set_tiling(&mut self, tiling: Tiling) {
        self.tiling = tiling;
        self.colors = tiling.quotient.orbit_size();
        self.view.reset_colors();
        self.view.settle(&self.tiling);
        self.window.as_ref().unwrap().set_title(&self.title());
    }

    fn paint(&mut self) {
        let out = self.gui.as_mut().unwrap().run(
            self.window.as_ref().unwrap(),
            &mut self.settings,
            &self.tiling,
            self.colors,
        );
        // The panel was laid out before any action; redraw to show it.
        let repaint_after = match out.action {
            Some(Action::SetTiling(t)) => {
                self.set_tiling(t);
                Duration::ZERO
            }
            Some(Action::ResetView) => {
                self.view = View::new();
                Duration::ZERO
            }
            None => out.repaint_after,
        };
        self.repaint_at = Instant::now().checked_add(repaint_after);
        let u = self.uniforms();
        self.renderer.as_mut().unwrap().render(&u, out.overlay);
    }

    fn key(&mut self, code: KeyCode) {
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
                self.set_tiling(t);
            }
            KeyCode::KeyL => {
                let Some(t) = self.tiling.with_ell_stepped(step.into()) else {
                    return;
                };
                self.set_tiling(t);
            }
            KeyCode::KeyC => self.settings.quotient_colors ^= true,
            KeyCode::KeyH => self.settings.show_panel ^= true,
            KeyCode::KeyR => self.view = View::new(),
            _ => return,
        }
        self.redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title(self.title()))
                .expect("create window"),
        );
        let renderer = Renderer::new(window.clone(), event_loop.owned_display_handle());
        self.gui = Some(Gui::new(&window, event_loop, renderer.max_texture_side()));
        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        // Not for egui, which would ask to repaint after every frame.
        if event == WindowEvent::RedrawRequested {
            self.paint();
            return;
        }
        let window = self.window.as_ref().unwrap();
        let egui = self.gui.as_mut().unwrap().on_window_event(window, &event);
        if egui.repaint {
            self.redraw();
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.renderer
                    .as_mut()
                    .unwrap()
                    .resize(size.width, size.height);
                self.redraw();
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
            } if !egui.consumed => self.key(code),
            WindowEvent::MouseInput {
                state: ElementState::Released,
                ..
            } => self.drag = None,
            WindowEvent::MouseInput { button, .. } if !egui.consumed => {
                self.drag = match button {
                    MouseButton::Left if self.modifiers.shift_key() => Some(Drag::Rotate),
                    MouseButton::Left
                        if self.cursor.is_some_and(|c| view::pannable(self.to_disk(c))) =>
                    {
                        Some(Drag::Pan)
                    }
                    MouseButton::Right => Some(Drag::Rotate),
                    _ => self.drag,
                };
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let (Some(drag), Some(old)) = (self.drag, self.cursor) {
                    let (from, to) = (self.to_disk(old), self.to_disk(position));
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
                    self.redraw();
                }
                self.cursor = Some(position);
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        match self.repaint_at {
            Some(t) if t <= Instant::now() => {
                self.repaint_at = None;
                self.redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Some(t) => event_loop.set_control_flow(ControlFlow::WaitUntil(t)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }
}

fn main() {
    println!("{}", ui::USAGE);
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.run_app(&mut App::new()).expect("run event loop");
}
