#![cfg_attr(nightly_portable_simd, feature(portable_simd))]

#[cfg(all(feature = "deterministic", feature = "native-fft"))]
compile_error!(
    "the deterministic and native-fft features are mutually exclusive; use --no-default-features --features deterministic"
);

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

mod decode;
mod numeric_plan;
mod optimized;
mod request;
mod spec;
mod vec_utils;

pub use decode::{
    Decoder, DecoderMetadata, DropOuts, FieldInfoEntry, LumaOutput, VitsMetrics, WriteableField,
    BLOCKSIZE,
};
pub use numeric_plan::CanonicalNumericPlanIdentity;
pub use request::{
    BoostBpf, BoostRampFilter, ColorSystem, DecodeOptions, DecodeProfile, DecodeRequest,
    DecoderParams, DeemphasisParams, FieldOrderAction, FmAudioChannels, LineSystem, LogisticParams,
    NonlinearParams, NotchFilter, RfPeaking, ShelfKind, SysParams, VideoBpf, VideoEqBand,
    VideoEqParams, VideoLumaFilter, WowInterpolation,
};
pub use spec::DecoderSpec;

pub type DeterministicHashMap<K, V> = HashMap<K, V, BuildHasherDefault<DefaultHasher>>;
