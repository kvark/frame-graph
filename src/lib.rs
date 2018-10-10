extern crate gfx_hal as hal;

struct Epoch(u16);

// TODO: split the public attachments to use slices, versus internal attachments to use vectors.
pub struct Attachments<S> {
    pub inputs: Vec<(ImageRef, S)>,
    pub outputs: Vec<(ImageRef, S, hal::pass::AttachmentOps)>,
    pub depth_stencil: Option<(ImageRef, S, hal::pass::AttachmentOps, hal::pass::AttachmentOps)>,
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


pub struct Buffer<'a, B: hal::Backend> {
    pub size: u64,
    pub stride: u64,
    pub body: Option<(&'a B::Buffer, hal::buffer::State)>,
}

trait Resource {
    type Part: Clone;
    type State;
    fn full_part(&self) -> Self::Part;
    fn idle_state(&self) -> Self::State;
}

impl<'a, B: hal::Backend> Resource for Buffer<'a, B> {
    type Part = ();
    type State = hal::buffer::State;

    fn full_part(&self) -> Self::Part {
        ()
    }
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
    type Part = hal::image::SubresourceRange;
    type State = hal::image::State;

    fn full_part(&self) -> Self::Part {
        hal::image::SubresourceRange {
            aspects: self.format.aspects(),
            levels: 0 .. self.kind.get_num_levels(),
            layers: 0 .. self.kind.get_num_layers(),
        }
    }
    fn idle_state(&self) -> Self::State {
        match self.body {
            Some((_, state)) => state,
            None => (hal::image::Access::empty(), hal::image::ImageLayout::Undefined),
        }
    }
}


struct TrackedState<R: Resource> {
    part: R::Part,
    state: R::State,
    /// The list of dependent epoch for this state,
    /// each with the respective sub-resource.
    /// Those parts have to be disjoint,
    /// and their union has to be a subset of `self.part`.
    dependencies: Vec<(Epoch, R::Part)>,
}

struct TrackedResource<R: Resource> {
    resource: R,
    states: Vec<TrackedState<R>>,
}

impl<R: Resource> From<R> for TrackedResource<R> {
    fn from(resource: R) -> Self {
        TrackedResource {
            states: vec![
                TrackedState {
                    part: resource.full_part(),
                    state: resource.idle_state(),
                    dependencies: Vec::new(),
                },
            ],
            resource,
        }
    }
}

enum Task<B: hal::Backend> {
    Graphics(Box<RenderTask<B>>, Attachments<Epoch>, Resources),
    Transfer(Box<WorkTask<B, hal::Transfer>>),
    Compute(Box<WorkTask<B, hal::Compute>>),
}

pub struct BufferRef(usize);
pub struct ImageRef(usize);


/*
impl SubresourceRange {
    /// Return true if the range contains no subresources.
    pub fn is_empty(&self) -> bool {
        self.aspects.is_empty() ||
        self.levels.start >= self.levels.end ||
        self.layers.start >= self.layers.end
    }
}


fn intersect_image_parts(impl BitAnd for SubresourceRange {
    type Output = Self;
    fn bitand(self, sl: SubresourceRange) -> Self {
        SubresourceRange {
            aspects: self.aspects & sl.aspects,
            //TODO: ensure the result ranges are at least valid
            levels: self.levels.start.max(sl.levels.start) .. self.levels.end.min(sl.levels.end),
            layers: self.layers.start.max(sl.layers.start) .. self.layers.end.min(sl.layers.end),
        }
    }
}
*/

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

    fn stamp_image(
        &mut self,
        image_ref: &ImageRef,
        subresource: hal::image::SubresourceRange,
        access: hal::image::Access,
        layout: hal::image::ImageLayout,
    ) -> Epoch {
        // for each resource
        //   if it's being written to, or the last use was writing to
        //     create a new resource epoch
        //     wrap up the previous one, make a transition
        //  if it's read and the last one was reading
        //     add the access flags to the current epoch
        let tr = &mut self.images[image_ref.0];
        // On the way down, we may end up with disjoint sub-sets of the original
        // sub-resource that we'll need to track independently.
        let mut active_parts = vec![subresource];
        for (i, ts) in tr.states.iter_mut().enumerate().rev() {
            let mut j = 0;
            while j < active_parts.len() {
                let part = &mut active_parts[j];
                let intersection = hal::image::SubresourceRange {
                    aspects: ts.part.aspects & part.aspects,
                    levels: ts.part.levels.start.max(part.levels.start) .. ts.part.levels.end.min(part.levels.end),
                    layers: ts.part.layers.start.max(part.layers.start) .. ts.part.layers.end.min(part.layers.end),
                };
                if intersection.aspects.is_empty() ||
                    intersection.levels.start >= intersection.levels.end ||
                    intersection.layers.start >= intersection.layers.end
                {
                    // this state doesn't affect our case
                    j += 1;
                    continue
                }
                if access.intersects(&hal::image::Access::ALL_WRITE) ||
                    ts.state.0.intersects(&hal::image::Access::ALL_WRITE)
                {
                    // dependency is required
                }
            }
        }
        panic!("Sub-parts {:?} are not covered by full range {:?}", active_parts, tr.resource.full_part());
    }

    pub fn add_render_task<T: RenderTask<B>>(
        &mut self,
        task: T,
        attachments: Attachments<(hal::image::SubresourceLayers, hal::image::ImageLayout)>,
        resources: Resources,
    ) {
        let mut atts = Attachments {
            inputs: Vec::with_capacity(attachments.inputs.len()),
            outputs: Vec::with_capacity(attachments.outputs.len()),
            depth_stencil: None,
        };
        for (ir, (sl, layout)) in attachments.inputs {
            let epoch = self.stamp_image(&ir, sl.into(), hal::image::Access::INPUT_ATTACHMENT_READ, layout);
            atts.inputs.push((ir, epoch));
        }
        for (ir, (sl, layout), ops) in attachments.outputs {
            let epoch = self.stamp_image(&ir, sl.into(), hal::image::Access::COLOR_ATTACHMENT_WRITE, layout);
            //TODO: transition from `Undefined` if the contents are not relevant according to the `ops`
            atts.outputs.push((ir, epoch, ops));
        }
        if let Some((ir, (sl, layout), depth_ops, stencil_ops)) = attachments.depth_stencil {
            let epoch = self.stamp_image(&ir, sl.into(), hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE, layout);
            atts.depth_stencil = Some((ir, epoch, depth_ops, stencil_ops));
        }
        //TODO: resources, etc
        self.tasks.push(Task::Graphics(Box::new(task) as Box<_>, atts, resources));
    }

    pub fn run(&mut self, device: &mut B::Device) {
        use hal::Device;

        let mut nodes = Vec::new();
        for task in &mut self.tasks {
            nodes.push(match *task {
                Task::Graphics(ref mut t, _, _) => {
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
                    t.prepare(device, hal::pass::Subpass {
                        main_pass: &renderpass,
                        index: 0, //TODO
                    });
                    Node::Graphics(vec![t.as_mut()], renderpass, extent, Vec::new())
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
