mod cli;
mod decode;
mod exact_verify;
mod fields_match;
mod flac;
mod http_source;
mod metadata;
mod os;
mod profiles;
mod reader;
mod writer;

fn main() {
    if let Err(error) = cli::run_cli() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
