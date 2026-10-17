mod gpu;
mod hyp;
mod tiling;

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use glam::DMat3;

use gpu::{Renderer, Uniforms};
use hyp::lorentz_inv;
use tiling::Tiling;

/// Gap between the disk and the nearer window edge, px.
const MARGIN_PX: f32 = 8.0;
const LINE_WIDTH_PX: f32 = 1.5;

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    tiling: Tiling,
    /// Tiling frame -> screen frame.
    m: DMat3,
}

impl App {
    fn uniforms(&self) -> Uniforms {
        let (w, h) = self.renderer.as_ref().unwrap().size();
        let (w, h) = (w as f32, h as f32);
        let view = [w, h, (0.5 * w.min(h) - MARGIN_PX).max(1.0), LINE_WIDTH_PX];
        Uniforms::new(lorentz_inv(self.m), &self.tiling, view)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Hyperbolic tiling"))
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
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let u = self.uniforms();
                self.renderer.as_mut().unwrap().render(&u);
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("create event loop");
    let mut app = App {
        window: None,
        renderer: None,
        tiling: Tiling::new(7, 3).unwrap(),
        m: DMat3::IDENTITY,
    };
    event_loop.run_app(&mut app).expect("run event loop");
}
