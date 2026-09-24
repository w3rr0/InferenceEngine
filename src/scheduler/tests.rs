use super::*;
use crate::memory::BlockAllocator;

#[test]
fn test_scheduler_prefill_and_decode() {
    let mut allocator = BlockAllocator::new(100, 16);
    let mut scheduler = Scheduler::new(2);

    scheduler.add_request(Request::new(1, vec![10, 20, 30]));
    scheduler.add_request(Request::new(2, vec![40; 20]));

    let prefill_batch = scheduler.next_batch(&mut allocator).unwrap();

    assert!(prefill_batch.is_prefill);
    assert_eq!(prefill_batch.request_ids, vec![1, 2]);
    assert_eq!(prefill_batch.seq_lens, vec![3, 20]);
    assert_eq!(prefill_batch.input_tokens[0], vec![10, 20, 30]);

    assert_eq!(prefill_batch.block_tables[0].len(), 1);
    assert_eq!(prefill_batch.block_tables[1].len(), 2);

    assert_eq!(scheduler.pending.len(), 0);
    assert_eq!(scheduler.active.len(), 2);

    scheduler.active[0].generated_tokens.push(99);
    scheduler.active[1].generated_tokens.push(88);

    let decode_batch = scheduler.next_batch(&mut allocator).unwrap();

    assert!(!decode_batch.is_prefill);
    assert_eq!(decode_batch.request_ids, vec![1, 2]);
    assert_eq!(decode_batch.seq_lens, vec![4, 21]);
    assert_eq!(decode_batch.input_tokens[0], vec![99]);
    assert_eq!(decode_batch.input_tokens[1], vec![88]);
}