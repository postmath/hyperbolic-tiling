mod gpu;

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use gpu::{Renderer, Uniforms};

/// Gap between the disk and the nearer window edge, px.
const MARGIN_PX: f32 = 8.0;
const LINE_WIDTH_PX: f32 = 1.5;

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
}

impl App {
    fn uniforms(&self) -> Uniforms {
        let (w, h) = self.renderer.as_ref().unwrap().size();
        let (w, h) = (w as f32, h as f32);
        Uniforms {
            view: [w, h, (0.5 * w.min(h) - MARGIN_PX).max(1.0), LINE_WIDTH_PX],
        }
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
    event_loop
        .run_app(&mut App::default())
        .expect("run event loop");
}
