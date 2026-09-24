use super::*;

#[test]
fn test_allocator_initialization() {
    let allocator = BlockAllocator::new(1024, 16);
    assert_eq!(allocator.available_blocks(), 1024);
    assert_eq!(allocator.total_blocks, 1024);
    assert_eq!(allocator.block_size, 16);
}

#[test]
fn test_allocate_multiple_success() {
    let mut allocator = BlockAllocator::new(100, 16);

    let table = allocator.allocate_multiple(3).expect("The allocation should succeed");

    assert_eq!(table.blocks.len(), 3);
    assert_eq!(table.blocks, vec![0, 1, 2]);
    assert_eq!(allocator.available_blocks(), 97);
}

#[test]
fn test_out_of_memory_handling() {
    let mut allocator = BlockAllocator::new(10, 16);

    let _table1 = allocator.allocate_multiple(8).unwrap();
    assert_eq!(allocator.available_blocks(), 2);

    let result = allocator.allocate_multiple(3);
    assert_eq!(result, Err(AllocatorError::OutOfMemory));

    assert_eq!(allocator.available_blocks(), 2);
}

#[test]
fn test_freeing_memory() {
    let mut allocator = BlockAllocator::new(10, 16);

    let table = allocator.allocate_multiple(4).unwrap();
    assert_eq!(allocator.available_blocks(), 6);

    allocator.free_table(&table);

    assert_eq!(allocator.available_blocks(), 10);
}

#[test]
fn test_allocate_single_block_for_decode() {
    let mut allocator = BlockAllocator::new(5, 16);

    let block = allocator.allocate_block().expect("Should get a single block");
    assert_eq!(block, 0);
    assert_eq!(allocator.available_blocks(), 4);
}