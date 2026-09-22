use block_manager::block_manager::BlockManager;

fn main() {
    let _manager = BlockManager::new(1024, 16);
    println!("Project compiles properly");
}