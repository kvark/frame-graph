extern crate gfx_core;

use gfx_core::{buffer, image, pass};


pub trait FrameBuilder: IntoIterator {
    type BufferRef;
    type ImageRef;

    fn buffer(&mut self) -> Self::BufferRef;
    fn image(&mut self) -> Self::ImageRef;

    fn pass(&mut self, Self::Item,
        Resources<Self::BufferRef, Self::ImageRef>,
        Option<Attachments<Self::ImageRef>>,
    );
}

pub struct Attachments<'a, Ref: 'a> {
    pub inputs: &'a [(&'a Ref, image::ImageLayout)],
    pub outputs: &'a [(&'a Ref, image::ImageLayout, pass::AttachmentOps)],
    pub depth_stencil: Option<(&'a Ref, image::ImageLayout)>,
    pub preserves: &'a [&'a Ref],
}

pub struct Resources<'a, BufferRef: 'a, ImageRef: 'a> {
    pub buffers: &'a [(&'a BufferRef, buffer::Access)],
    pub images: &'a [(&'a ImageRef, image::Access, image::ImageLayout)],
}
