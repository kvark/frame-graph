extern crate gfx_hal as hal;


struct Epoch(u16);

pub struct Attachments {
    pub inputs: Vec<(ImageRef, hal::image::SubresourceLayers, hal::image::ImageLayout)>,
    pub outputs: Vec<(ImageRef, hal::image::SubresourceLayers, hal::image::ImageLayout, hal::pass::AttachmentOps)>,
    pub depth_stencil: Option<(ImageRef, hal::image::SubresourceLayers, hal::image::ImageLayout, hal::pass::AttachmentOps, hal::pass::AttachmentOps)>,
}

pub struct Resources {
    pub buffers: Vec<(BufferRef, hal::buffer::Access)>,
    pub images: Vec<(ImageRef, hal::image::SubresourceRange, hal::image::Access, hal::image::ImageLayout)>,
}

/// A render task in a frame graph.
///
/// It is guaranteed that the render sub-pass encoder provided to `record` method
/// is compatible with the sub-pass given to the last call to `prepare`.
pub trait RenderTask<B: hal::Backend>: 'static {
    /// Prepare the pass by recompiling all the pipeline states to work in a given sub-pass.
    fn prepare<'a>(&mut self, &mut B::Device, hal::pass::Subpass<'a, B>);
    /// Record the graphics work to a given sub-pass encoder.
    fn record<'a>(&self, &mut hal::command::RenderSubpassCommon<'a, B>);
}

pub trait WorkTask<B: hal::Backend, C> {
    /// Record the work to a given command buffer.
    fn record<'a>(&self, &mut hal::command::CommandBuffer<'a, B, C>);
}

enum Task<B: hal::Backend> {
    Graphics(Vec<(Box<RenderTask<B>>, Attachments, Resources)>),
    Transfer(Box<WorkTask<B, hal::Transfer>>),
    Compute(Box<WorkTask<B, hal::Compute>>),
}


pub struct Buffer<'a, B: hal::Backend> {
    pub size: u64,
    pub stride: u64,
    pub body: Option<(&'a B::Buffer, hal::buffer::State)>,
}

trait Resource {
    type State;
    fn idle_state(&self) -> Self::State;
}

impl<'a, B: hal::Backend> Resource for Buffer<'a, B> {
    type State = hal::buffer::State;
    fn idle_state(&self) -> Self::State {
        match self.body {
            Some((_, state)) => state,
            None => hal::buffer::State::empty(),
        }
    }
}

pub struct Image<'a, B: hal::Backend> {
    pub kind: hal::image::Kind,
    pub format: hal::format::Format,
    pub body: Option<(&'a B::Image, hal::image::State)>,
}

impl<'a, B: hal::Backend> Resource for Image<'a, B> {
    type State = hal::image::State;
    fn idle_state(&self) -> Self::State {
        match self.body {
            Some((_, state)) => state,
            None => (hal::image::Access::empty(), hal::image::ImageLayout::Undefined),
        }
    }
}


struct TrackedState<S> {
    state: S,
    previous_epoch: Epoch,
}

struct TrackedResource<R: Resource> {
    resource: R,
    states: Vec<TrackedState<R::State>>,
}

impl<R: Resource> From<R> for TrackedResource<R> {
    fn from(resource: R) -> Self {
        TrackedResource {
            states: vec![
                TrackedState {
                    state: resource.idle_state(),
                    previous_epoch: Epoch(0),
                },
            ],
            resource,
        }
    }
}

pub struct BufferRef(usize);
pub struct ImageRef(usize);


pub struct FrameGraph<'a, B: hal::Backend> {
    buffers: Vec<TrackedResource<Buffer<'a, B>>>,
    images: Vec<TrackedResource<Image<'a, B>>>,
    tasks: Vec<Task<B>>,
    //memory_pool: Vec<B::Memory>,
    queue: hal::CommandQueue<B, hal::General>,
    command_pool: hal::CommandPool<B, hal::General>,
    frame_fence: B::Fence,
}

enum Node<'a, B: hal::Backend> {
    Graphics(Vec<&'a RenderTask<B>>, B::RenderPass, hal::device::Extent, Vec<hal::command::ClearValue>),
    Transfer(&'a WorkTask<B, hal::Transfer>),
    Compute(&'a WorkTask<B, hal::Compute>),
}

impl<'a, B: hal::Backend> FrameGraph<'a, B> {
    pub fn use_buffer(&mut self, buffer: Buffer<'a, B>) -> BufferRef {
        let id = self.buffers.len();
        self.buffers.push(buffer.into());
        BufferRef(id)
    }

    pub fn use_image(&mut self, image: Image<'a, B>) -> ImageRef {
        let id = self.images.len();
        self.images.push(image.into());
        ImageRef(id)
    }

    pub fn add_render_task<T: RenderTask<B>>(
        &mut self,
        task: T,
        attachments: Attachments,
        resources: Resources,
    ) {
        let pass = (Box::new(task) as Box<_>, attachments, resources);
        self.tasks.push(Task::Graphics(vec![pass]));
        // for each resource
        //   if it's being written to, or the last use was writing to
        //     create a new resource epoch
        //     wrap up the previous one, make a transition
        //  if it's read and the last one was reading
        //     add the access flags to the current epoch
    }

    pub fn run(&mut self, device: &mut B::Device) {
        use hal::Device;

        let mut nodes = Vec::new();
        for task in &mut self.tasks {
            nodes.push(match *task {
                Task::Graphics(ref mut task_passes) => {
                    //TODO: combine multiple passes into fewer nodes
                    let attachments = vec![];
                    let subpasses = vec![];
                    let dependencies = vec![];
                    let renderpass = device.create_render_pass(&attachments, &subpasses, &dependencies);
                    let extent = hal::device::Extent { //TODO
                        width: 0,
                        height: 0,
                        depth: 0,
                    };
                    let subpasses = task_passes
                        .iter_mut()
                        .enumerate()
                        .map(|(i, t)| {
                            t.0.prepare(device, hal::pass::Subpass {
                                main_pass: &renderpass,
                                index: i as _,
                            });
                            t.0.as_ref()
                        })
                        .collect();
                    Node::Graphics(subpasses, renderpass, extent, Vec::new())
                }
                Task::Transfer(ref t) => {
                    Node::Transfer(t.as_ref())
                }
                Task::Compute(ref t) => {
                    Node::Compute(t.as_ref())
                }
            });
        }

        //TODO: allocation of command buffers should be dynamic as needed
        // submissions could be many, given the semaphores controlling the order.
        let mut cbuf = self.command_pool.acquire_command_buffer(false);
        for node in &nodes {
            match *node {
                Node::Graphics(ref tasks, ref pass, extent, ref clear_values) => {
                    let attachments = vec![]; //TODO
                    let framebuffer = device.create_framebuffer(pass, &attachments, extent)
                        .unwrap();
                    let area = hal::command::Rect {
                        x: 0,
                        y: 0,
                        w: extent.width as _,
                        h: extent.height as _,
                    };
                    let mut encoder = cbuf.begin_renderpass_inline(
                        pass,
                        &framebuffer,
                        area,
                        clear_values,
                    );
                    for task in tasks {
                        task.record(&mut encoder);
                        encoder = encoder.next_subpass_inline();
                    }
                }
                Node::Transfer(ref task) => {
                    task.record(cbuf.downgrade());
                }
                Node::Compute(ref task) => {
                    task.record(cbuf.downgrade());
                }
            }
        }

        let submit = cbuf.finish();
        let submission = hal::Submission::new()
            .submit(Some(submit));
        device.reset_fences(&[&self.frame_fence]);
        self.queue.submit(submission, Some(&self.frame_fence));
    }
}
