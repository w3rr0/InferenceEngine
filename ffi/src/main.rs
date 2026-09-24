use pyo3::prelude::*;
use std::io::Write;

use block_manager::block_manager::BlockManager;

fn main() -> PyResult<()> {
    // VRAM management initialization
    let mut manager = BlockManager::new(1024, 16);

    // Simulation of an incoming client request (prefill phase)
    let seq_id = 1;
    let mut current_seq_len = 15;

    manager.allocate(seq_id, current_seq_len).expect("VRAM allocation error");
    println!("[BlockManager] Blocked memory for prompt. Available blocks: {}", manager.available_blocks());

    // Simulation of prompt tokenization
    let mut current_input_ids = vec![vec![312]];
    let mut generated_tokens = Vec::new();

    println!("[FFI] Launching the Python environment");

    Python::attach(|py| -> PyResult<()> {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (".",))?;

        if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
            let sysconfig = py.import("sysconfig")?;

            let vars = pyo3::types::PyDict::new(py);
            vars.set_item("base", &venv)?;
            vars.set_item("platbase", &venv)?;

            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("vars", vars)?;

            let site_packages = sysconfig.call_method("get_path", ("purelib",), Some(&kwargs))?;
            path.call_method1("append", (site_packages,))?;
        }

        // Import from python
        let py_module = PyModule::import(py, "model_management.model_management")?;

        let model = py_module.call_method0("load_engine")?;
        println!("[FFI] Model pomyślnie załadowany do pamięci. Rozpoczynam generację.\n");

        // Decode phase
        for _step in 1..=10 {
            let block_table = manager.get_block_table(seq_id).unwrap().clone();

            let block_tables = vec![block_table];
            let seq_lens = vec![current_seq_len];

            let args = (
                &model,
                current_input_ids.clone(),
                block_tables,
                seq_lens,
            );

            let next_tokens_py = py_module.call_method1("forward_step", args)?;

            // Extracting generated tokens back to Rust
            let next_tokens: Vec<usize> = next_tokens_py.extract()?;
            let new_token = next_tokens[0];

            generated_tokens.push(new_token);
            print!("{} ", new_token);
            std::io::stdout().flush().unwrap();

            current_input_ids = vec![vec![new_token]];
            current_seq_len += 1;

            let allocated_new_block = manager.append_token(seq_id).expect("No memory left in VRAM");
            if allocated_new_block {
                println!("\n[BlockManager] The request exceeded capacity. A new VRAM block was allocated. Available blocks: {}", manager.available_blocks());
            }
        }

        Ok(())
    })?;

    println!("\n\n[Engine] Generation complete");
    println!("[Engine] Full sequence of output tokens: {:?}", generated_tokens);

    // Closing the connection and freeing memory
    manager.free_sequence(seq_id).expect("Memory deallocation error");
    println!("[BlockManager] Request resources released. Available VRAM blocks: {}", manager.available_blocks());

    Ok(())
}
