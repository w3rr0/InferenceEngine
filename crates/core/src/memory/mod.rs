pub mod block;
pub mod allocator;

#[cfg(test)]
mod tests;

pub use block::{BlockId, BlockTable};
pub use allocator::{AllocatorError, BlockAllocator};