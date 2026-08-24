use std::env;
use std::process::Command;

fn has_codegen_option(arguments: &[&str], option: &str) -> bool {
    arguments.windows(2).any(|pair| pair == ["-C", option])
        || arguments
            .iter()
            .any(|argument| argument.strip_prefix("-C") == Some(option))
}

fn require_deterministic_codegen_contract() {
    let encoded = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let arguments = encoded
        .split('\u{1f}')
        .filter(|argument| !argument.is_empty())
        .collect::<Vec<_>>();
    let required = [
        "no-vectorize-loops",
        "no-vectorize-slp",
        "llvm-args=--fp-contract=off",
    ];
    let missing = required
        .iter()
        .copied()
        .filter(|option| !has_codegen_option(&arguments, option))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        panic!(
            "the deterministic feature requires strict code generation; missing RUSTFLAGS options: {}. Use RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp -C llvm-args=--fp-contract=off'",
            missing.join(", ")
        );
    }
    if has_codegen_option(&arguments, "target-cpu=native") {
        panic!(
            "the deterministic feature rejects -C target-cpu=native; use the target's documented baseline CPU contract"
        );
    }
}

// Detect a nightly toolchain so the portable SIMD kernels can be gated on
// it; stable builds fall back to the scalar paths.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(nightly_portable_simd)");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let is_nightly = Command::new(rustc)
        .arg("--version")
        .output()
        .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains("nightly"));
    let deterministic = env::var_os("CARGO_FEATURE_DETERMINISTIC").is_some();
    if deterministic {
        require_deterministic_codegen_contract();
    }
    if is_nightly && !deterministic {
        println!("cargo:rustc-cfg=nightly_portable_simd");
    }
}
