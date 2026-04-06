use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let engine_build_dir = PathBuf::from(&manifest_dir).join("../engine/build/Release");

    // 1. Tell Cargo where to find the .lib file for compile-time linking
    println!("cargo:rustc-link-search=native={}", engine_build_dir.display());
    println!("cargo:rustc-link-lib=engine");

    // --- THE CI AUTOMATION FIX ---
    
    // 2. Cargo gives us an environment variable called OUT_DIR 
    // (e.g., target/debug/build/manager-xxxx/out)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // 3. We navigate up three levels to get to the root `target/debug/` folder
    // where the actual compiled test .exe files are placed.
    let target_dir = out_dir.parent().unwrap().parent().unwrap().parent().unwrap();

    let dll_source = engine_build_dir.join("engine.dll");
    let dll_dest = target_dir.join("engine.dll");

    // 4. Automatically copy the DLL over every time we compile!
    // We only execute the copy if the source DLL actually exists.
    if dll_source.exists() {
        fs::copy(&dll_source, &dll_dest)
            .expect("Failed to copy engine.dll to target directory");
    } else {
        println!("cargo:warning=engine.dll not found at {:?}. Did you compile the C++ engine?", dll_source);
    }
}