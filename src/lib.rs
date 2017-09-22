extern crate gfx_core;

use gfx_core::{buffer, image, pass};
use gfx_core::format::Format;
use gfx_core::Backend;


pub trait FrameBuilder<B: Backend>: IntoIterator {
    type BufferRef;
    type ImageRef;

    fn buffer(&mut self, size: u64, stride: u64, Option<&B::Buffer>) -> Self::BufferRef;
    fn image(&mut self, image::Kind, Format, Option<&B::Image>) -> Self::ImageRef;

    fn pass(&mut self, Self::Item,
        Resources<Self::BufferRef, Self::ImageRef>,
        Option<Attachments<Self::ImageRef>>,
    );
}

pub struct Attachments<'a, Ref: 'a> {
    pub inputs: &'a [(&'a Ref, image::SubresourceLayers, image::ImageLayout)],
    pub outputs: &'a [(&'a Ref, image::SubresourceLayers, image::ImageLayout, pass::AttachmentOps)],
    pub depth_stencil: Option<(&'a Ref, image::SubresourceLayers, image::ImageLayout, pass::AttachmentOps, pass::AttachmentOps)>,
    pub preserves: &'a [(&'a Ref, image::SubresourceLayers)],
}

pub struct Resources<'a, BufferRef: 'a, ImageRef: 'a> {
    pub buffers: &'a [(&'a BufferRef, buffer::Access)],
    pub images: &'a [(&'a ImageRef, image::SubresourceRange, image::Access, image::ImageLayout)],
}
