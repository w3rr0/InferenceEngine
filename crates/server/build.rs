fn main() -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: build.rs runs before anything else in this crate, single-threaded - Nothing else reads the environment concurrently at this point
    unsafe {
        std::env::set_var(
            "PROTOC",
            protoc_bin_vendored::protoc_bin_path()?,
        );
    }

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&["proto/generation.proto"], &["proto"])?;
    Ok(())
}