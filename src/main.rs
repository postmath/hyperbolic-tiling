mod gpu;
mod hyp;
mod tiling;
mod view;

use std::sync::Arc;

use glam::DVec2;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};

use gpu::{Renderer, Uniforms};
use hyp::lorentz_inv;
use tiling::Tiling;
use view::View;

/// Gap between the disk and the nearer window edge, px.
const MARGIN_PX: f32 = 8.0;
const LINE_WIDTH_PX: f32 = 1.5;

const USAGE: &str = "\
Drag: pan. Right drag or Shift+drag: rotate.
P / Shift+P: step p up / down. Q / Shift+Q: step q. R: reset view.";

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Pan,
    Rotate,
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    tiling: Tiling,
    view: View,
    cursor: Option<PhysicalPosition<f64>>,
    modifiers: ModifiersState,
    drag: Option<Drag>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            tiling: Tiling::new(7, 3).unwrap(),
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
        Uniforms::new(
            lorentz_inv(self.view.m),
            &self.tiling,
            [w, h, r, LINE_WIDTH_PX],
        )
    }

    fn redraw(&self) {
        self.window.as_ref().unwrap().request_redraw();
    }

    fn title(&self) -> String {
        format!("{{{},{}}} hyperbolic tiling", self.tiling.p, self.tiling.q)
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
                self.tiling = t;
                self.view.settle(&self.tiling);
                self.window.as_ref().unwrap().set_title(&self.title());
            }
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
        self.renderer = Some(Renderer::new(
            window.clone(),
            event_loop.owned_display_handle(),
        ));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.renderer
                    .as_mut()
                    .unwrap()
                    .resize(size.width, size.height);
                self.redraw();
            }
            WindowEvent::RedrawRequested => {
                let u = self.uniforms();
                self.renderer.as_mut().unwrap().render(&u);
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
            } => self.key(code),
            WindowEvent::MouseInput { state, button, .. } => {
                self.drag = match (state, button) {
                    (ElementState::Pressed, MouseButton::Left) if self.modifiers.shift_key() => {
                        Some(Drag::Rotate)
                    }
                    (ElementState::Pressed, MouseButton::Left) => Some(Drag::Pan),
                    (ElementState::Pressed, MouseButton::Right) => Some(Drag::Rotate),
                    (ElementState::Released, _) => None,
                    _ => self.drag,
                };
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let (Some(drag), Some(old)) = (self.drag, self.cursor) {
                    let (from, to) = (self.to_disk(old), self.to_disk(position));
                    match drag {
                        Drag::Pan => self.view.pan(from, to),
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
}

fn main() {
    println!("{USAGE}");
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.run_app(&mut App::new()).expect("run event loop");
}
