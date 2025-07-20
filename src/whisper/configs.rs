use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::NonZeroUsize;
use std::str::FromStr;

use strum::{AsRefStr, Display, EnumCount, EnumIter, EnumString, FromRepr, IntoStaticStr};
use whisper_rs;

use crate::whisper::model::{DefaultModelType, Model, ModelId};

// Store an ID instead of the actual model -- come back to implement this once Model is refactored.

/// Versioned wrapper for supported whisper-realtime configuration types.
/// Can be serialized (using serde or otherwise) to persist settings.
/// V1: Legacy, whisper and realtime configurations.
/// V2: Current whisper-only configurations.
/// RealtimeV1: A complete realtime-configuration. Composed of V2 and RealtimeConfigs
///
/// NOTE: RealtimeConfigs is not included directly; it cannot be used alone for transcription
/// NOTE: Until WhisperConfigsV2 is stabilized, it is not reccomended to serialize either:
/// WhisperConfigsV2, WhisperRealtimeConfigs.
/// Instead, serialize WhisperConfigsV1, clone and consume it to pass into a Transcriber object
/// as necessary.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub enum Configs {
    V1(WhisperConfigsV1),
    V2(WhisperConfigsV2),
    RealtimeV1(WhisperRealtimeConfigs),
}

impl Configs {
    /// Used to migrate (older) V1 configs to RealtimeV1 configs when using a different hasher to
    /// compute the ModelId. This is to preserve as much information as possible from the legacy
    /// implementation. You are not restricted to using hashes for the ModelId and can set
    /// the value in your configs as you see fit.
    ///
    /// This method only affects V1; V2 and RealtimeV1 will just return the same object, as they already
    /// have ways to change the stored model ID for key-value lookup.
    ///
    /// If only V2 configurations are needed, chain with .into_v2() after calling this function.
    pub fn migrate_v1_with_hasher<H: Hasher>(self, hasher: &mut H) -> Self {
        match self {
            Configs::V1(v1) => Self::RealtimeV1(v1.into_realtime_v1_with_hasher(hasher)),
            Configs::V2(_) => self,
            Configs::RealtimeV1(_) => self,
        }
    }

    /// Converts configuration to WhisperRealtimeConfigurations (RealtimeV1).
    /// Realtime configurations are preserved from V1.
    /// If called on V2, RealtimeConfigurations::default() is supplied
    pub fn into_realtime_v1(self) -> Self {
        match self {
            Configs::V1(v1) => Self::RealtimeV1(v1.into_realtime_v1()),
            Configs::V2(v2) => Self::RealtimeV1(v2.into_realtime_v1()),
            Configs::RealtimeV1(_) => self,
        }
    }

    /// Converts configuration to WhisperConfigsV2.
    /// Note: Migrating from V1 or RealtimeV1 to V2 is lossy; realtime configurations are not part
    /// of WhisperConfigs V2
    pub fn into_v2(self) -> Self {
        match self {
            Configs::V1(v1) => Self::V2(v1.into_v2()),
            Configs::V2(_) => self,
            Configs::RealtimeV1(r_v1) => Self::V2(r_v1.into_whisper_v2_configs()),
        }
    }
}

/// A configurations type that holds a subset of useful configurations for whisper-rs::FullParams and whisper-rs::WhisperContextParams,
/// a transcription model, and a flag to indicate whether the GPU should be used to run the transcription.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug)]
pub struct WhisperConfigsV2 {
    // Whisper FullParams data
    /// The number of threads to use during whisper transcription. Defaults follow [whisper_rs::FullParams::set_n_threads]
    n_threads: std::ffi::c_int,
    /// The number of past text tokens to use as a prompt for the decoder. Defaults follow [whisper_rs::FullParams::set_n_max_text_ctx]
    max_past_prompt_tokens: std::ffi::c_int,
    /// The whisper sampling strategy: Greedy or BeamSearch
    sampling_strategy: WhisperSamplingStrategy,
    /// Translate the output to the language specified by self::language
    translate: bool,
    /// The target output language. Set to auto or None to auto-detect. Defaults follow [whisper_rs::FullParams::set_language]
    language: Option<Language>,
    /// Prevent using previous context as an initial prompt for the decoder.
    use_no_context: bool,
    // Whisper Context data
    /// An id for grabbing the model before transcription.
    model_id: Option<ModelId>,
    /// Use the gpu during transcription
    use_gpu: bool,
    /// Use flash attention
    flash_attention: bool,
}

impl WhisperConfigsV2 {
    pub fn new() -> Self {
        Self {
            n_threads: 1,
            max_past_prompt_tokens: 0,
            sampling_strategy: WhisperSamplingStrategy::Greedy { best_of: 1 },
            translate: false,
            language: None,
            use_gpu: false,
            flash_attention: false,
            use_no_context: false,
            model_id: None,
        }
    }

    /// Sets the number of threads. This cannot be zero and will always be set to a minimum of 1 thread.
    pub fn with_n_threads(mut self, num_threads: usize) -> Self {
        let threads = num_threads.max(1);
        self.n_threads = threads as std::ffi::c_int;
        self
    }

    /// Sets the max number of past tokens used to prompt the next decode
    pub fn with_max_past_prompt_tokens(mut self, num_prompt_tokens: usize) -> Self {
        self.max_past_prompt_tokens = num_prompt_tokens as std::ffi::c_int;
        self
    }

    /// Toggles translating from the input language to english.
    pub fn set_translate(mut self, set_translate: bool) -> Self {
        self.translate = set_translate;
        self
    }

    /// Sets the output language
    pub fn with_language(mut self, language: Option<Language>) -> Self {
        self.language = language;
        self
    }

    /// Toggles whether to use the gpu to accelerate transcription.
    /// For realtime applications, this should always be set true.
    pub fn set_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Toggles preventing using past context in the decoder.
    /// For realtime applications, this should always be set true.
    pub fn set_use_no_context(mut self, use_no_context: bool) -> Self {
        self.use_no_context = use_no_context;
        self
    }
    /// Toggles using flash attention when supported.
    /// For realtime applications, this should always be set true.
    pub fn set_flash_attention(mut self, flash_attention: bool) -> Self {
        self.flash_attention = flash_attention;
        self
    }

    /// Sets the sampling strategy.
    pub fn with_sampling_strategy(mut self, sampling_strategy: WhisperSamplingStrategy) -> Self {
        self.sampling_strategy = sampling_strategy;
        self
    }

    /// Sets the handle to a model.
    pub fn with_model_id(mut self, model_id: Option<ModelId>) -> Self {
        self.model_id = model_id;
        self
    }

    /// Gets the number of threads used in transcription.
    pub fn n_threads(&self) -> usize {
        self.n_threads as usize
    }

    /// Gets the max number of past tokens used as prompt for the decoder
    pub fn max_past_prompt_tokens(&self) -> usize {
        self.max_past_prompt_tokens as usize
    }

    /// Indicates whether whisper is set to translate the output text to the specified output language
    pub fn translate(&self) -> bool {
        self.translate
    }
    /// Indicates the selected output language. None = auto = Auto-detect language.
    pub fn language(&self) -> &Option<Language> {
        &self.language
    }

    /// Indicates whether the gpu should be used.
    pub fn using_gpu(&self) -> bool {
        self.use_gpu
    }

    /// Indicates whether past context is used to prompt the decoder.
    pub fn using_no_context(&self) -> bool {
        self.use_no_context
    }
    /// Indicates whether flash attention is being used
    pub fn using_flash_attention(&self) -> bool {
        self.flash_attention
    }

    /// Borrows the handle to a retrievable model.
    pub fn model_id(&self) -> &Option<ModelId> {
        &self.model_id
    }

    /// Consumes the configurations to convert to WhisperRealtime configs
    pub fn into_realtime_v1(self) -> WhisperRealtimeConfigs {
        WhisperRealtimeConfigs::new().with_whisper_configs(self)
    }

    /// Constructs a FullParams object ready to be passed to [whisper_rs::WhisperState::full]
    /// Note: these configurations do not cover FullParams in entirety
    /// Features are exposed on an as-needed bases.
    /// See: [whisper_rs::FullParams] for documentation.
    pub fn to_whisper_full_params(&self) -> whisper_rs::FullParams {
        let mut params = match self.sampling_strategy {
            WhisperSamplingStrategy::Greedy { best_of } => {
                whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy {
                    best_of: best_of as std::ffi::c_int,
                })
            }
            WhisperSamplingStrategy::BeamSearch {
                beam_size,
                patience,
            } => whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::BeamSearch {
                beam_size: beam_size as std::ffi::c_int,
                patience: patience as std::ffi::c_float,
            }),
        };

        params.set_n_threads(self.n_threads);
        params.set_n_max_text_ctx(self.max_past_prompt_tokens);
        params.set_translate(self.translate);
        if let Some(lang) = self.language {
            params.set_language(Some(lang.into()))
        }
        params.set_no_context(self.use_no_context);
        // Explicitly disable printing to stdout
        // These are considered to be unnecessary features and are unlikely to be exposed.
        // Progress can be obtained via the callback API
        // Timestamps will be added to offline transcription when needed.
        params.set_print_realtime(false);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params
    }

    /// Constructs a WhisperContextParameters object used to build [whisper_rs::WhisperContext]
    pub fn to_whisper_context_params(&self) -> whisper_rs::WhisperContextParameters {
        let mut params = whisper_rs::WhisperContextParameters::default();
        params.use_gpu(self.use_gpu);
        params.flash_attn(self.flash_attention);
        params
    }
}

impl Default for WhisperConfigsV2 {
    fn default() -> Self {
        let n_threads = std::cmp::min(
            4,
            std::thread::available_parallelism()
                .unwrap_or(NonZeroUsize::new(2).unwrap())
                .get(),
        );

        Self::new()
            .with_n_threads(n_threads)
            .with_max_past_prompt_tokens(MAX_PROMPT_TOKENS)
            .with_sampling_strategy(WhisperSamplingStrategy::Greedy { best_of: 1 })
            .set_gpu(cfg!(feature = "_gpu"))
    }
}

/// Encapsulates the whisper sampling strategy.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug)]
pub enum WhisperSamplingStrategy {
    Greedy { best_of: usize },
    BeamSearch { beam_size: usize, patience: f32 },
}

/// Encapsulates relevant configurations for tweaking realtime transcription.
/// All timeouts/audio lengths are measured in milliseconds
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug)]
pub struct RealtimeConfigs {
    realtime_timeout: u128,
    audio_sample_len: usize,
    vad_sample_len: usize,
}

impl RealtimeConfigs {
    pub fn new() -> Self {
        Self {
            realtime_timeout: 0,
            audio_sample_len: 0,
            vad_sample_len: 0,
        }
    }
    /// Sets the realtime timeout. Set to 0 for "Infinite"
    pub fn with_realtime_timeout(mut self, realtime_timeout: u128) -> Self {
        self.realtime_timeout = realtime_timeout;
        self
    }
    /// Sets the size of the audio sampling window. Defaults to 10 seconds (10 000 ms).
    pub fn with_audio_sample_len(mut self, len_ms: usize) -> Self {
        self.audio_sample_len = len_ms;
        self
    }
    /// Sets the size of the voice-detection sampling window. Defaults to 300ms.
    pub fn with_vad_sample_len(mut self, len_ms: usize) -> Self {
        self.vad_sample_len = len_ms;
        self
    }

    /// Gets the realtime timeout.
    pub fn realtime_timeout(&self) -> u128 {
        self.realtime_timeout
    }

    /// Gets the audio sampling window size.
    pub fn audio_sample_len(&self) -> usize {
        self.audio_sample_len
    }
    /// Gets the voice-detection sampling window size.
    pub fn vad_sample_len(&self) -> usize {
        self.vad_sample_len
    }
}

impl Default for RealtimeConfigs {
    fn default() -> Self {
        Self::new()
            // 1 hour
            .with_realtime_timeout(REALTIME_AUDIO_TIMEOUT)
            // 10 seconds / 10 000 ms
            .with_audio_sample_len(AUDIO_SAMPLE_MS)
            // .3 seconds / 300 ms
            .with_vad_sample_len(VAD_SAMPLE_MS)
    }
}

/// A Serializable enumeration that maps to ISO-639-1 format.
/// For use in [WhisperConfigsV2]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(
    Copy,
    Clone,
    Debug,
    PartialOrd,
    PartialEq,
    AsRefStr,
    IntoStaticStr,
    FromRepr,
    Display,
    EnumString,
    EnumIter,
    EnumCount,
)]
#[strum(serialize_all = "lowercase")]
pub enum Language {
    /// Automatically detect language
    Auto,
    En,
    Zh,
    De,
    Es,
    Ru,
    Ko,
    Fr,
    Ja,
    Pt,
    Tr,
    Pl,
    Ca,
    Nl,
    Ar,
    Sv,
    It,
    Id,
    Hi,
    Fi,
    Vi,
    He,
    Uk,
    El,
    Ms,
    Cs,
    Ro,
    Da,
    Hu,
    Ta,
    No,
    Th,
    Ur,
    Hr,
    Bg,
    Lt,
    La,
    Mi,
    Ml,
    Cy,
    Sk,
    Te,
    Fa,
    Lv,
    Bn,
    Sr,
    Az,
    Sl,
    Kn,
    Et,
    Mk,
    Br,
    Eu,
    Is,
    Hy,
    Ne,
    Mn,
    Bs,
    Kk,
    Sq,
    Sw,
    Gl,
    Mr,
    Pa,
    Si,
    Km,
    Sn,
    Yo,
    So,
    Af,
    Oc,
    Ka,
    Be,
    Tg,
    Sd,
    Gu,
    Am,
    Yi,
    Lo,
    Uz,
    Fo,
    Ht,
    Ps,
    Tk,
    Nn,
    Mt,
    Sa,
    Lb,
    My,
    Bo,
    Tl,
    Mg,
    As,
    Tt,
    Haw,
    Ln,
    Ha,
    Ba,
    Jw,
    Su,
    Yue,
}

/// A composite configurations type holding all relevant configurations for running realtime
/// transcription.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug)]
pub struct WhisperRealtimeConfigs {
    whisper: WhisperConfigsV2,
    realtime: RealtimeConfigs,
}

impl WhisperRealtimeConfigs {
    pub fn new() -> Self {
        Self {
            whisper: WhisperConfigsV2::new(),
            realtime: RealtimeConfigs::new(),
        }
    }

    // Convenience builder delegates
    /// Sets the whisper configurations.
    pub fn with_whisper_configs(mut self, w_configs: WhisperConfigsV2) -> Self {
        self.whisper = w_configs;
        self
    }
    /// Sets the realtime configurations.
    pub fn with_realtime_configs(mut self, r_configs: RealtimeConfigs) -> Self {
        self.realtime = r_configs;
        self
    }

    /// Sets the number of threads. This cannot be zero and will always be set to a minimum of 1 thread.
    pub fn with_n_threads(mut self, num_threads: usize) -> Self {
        let threads = num_threads.max(1);
        self.whisper.n_threads = threads as std::ffi::c_int;
        self
    }

    /// Sets the max number of past tokens used to prompt the next decode
    pub fn with_max_past_prompt_tokens(mut self, num_prompt_tokens: usize) -> Self {
        self.whisper.max_past_prompt_tokens = num_prompt_tokens as std::ffi::c_int;
        self
    }
    /// Toggles translating to the specified output language
    pub fn set_translate(mut self, set_translate: bool) -> Self {
        self.whisper.translate = set_translate;
        self
    }

    /// Sets the output language
    pub fn with_language(mut self, language: Option<Language>) -> Self {
        self.whisper.language = language;
        self
    }

    /// Toggles whether to use the gpu to accelerate transcription.
    /// For realtime applications, this should always be set true.
    pub fn set_gpu(mut self, use_gpu: bool) -> Self {
        self.whisper.use_gpu = use_gpu;
        self
    }
    /// Toggles preventing using past context in the decoder.
    /// For realtime applications, this should always be set true.
    pub fn set_use_no_context(mut self, use_no_context: bool) -> Self {
        self.whisper.use_no_context = use_no_context;
        self
    }

    /// Toggles using flash attention when supported.
    /// For realtime applications, this should always be set true.
    pub fn set_flash_attention(mut self, flash_attention: bool) -> Self {
        self.whisper.flash_attention = flash_attention;
        self
    }

    /// Sets the sampling strategy.
    pub fn with_sampling_strategy(mut self, sampling_strategy: WhisperSamplingStrategy) -> Self {
        self.whisper.sampling_strategy = sampling_strategy;
        self
    }

    /// Sets the model.
    pub fn with_model_id(mut self, model: Option<ModelId>) -> Self {
        self.whisper.model_id = model;
        self
    }

    /// Sets the model.
    pub fn with_realtime_timeout(mut self, realtime_timeout: u128) -> Self {
        self.realtime.realtime_timeout = realtime_timeout;
        self
    }

    /// Sets the size of the audio sampling window (in ms). Defaults to 10 seconds (10 000 ms).
    pub fn with_audio_sample_len(mut self, len_ms: usize) -> Self {
        self.realtime.audio_sample_len = len_ms;
        self
    }
    /// Sets the size of the voice-detection sampling window (in ms). Defaults to 300ms.
    pub fn with_vad_sample_len(mut self, len_ms: usize) -> Self {
        self.realtime.vad_sample_len = len_ms;
        self
    }

    // Whisper accessors
    /// Gets the number of threads used in transcription.
    pub fn n_threads(&self) -> usize {
        self.whisper.n_threads as usize
    }

    /// Gets the max number of past tokens used as prompt for the decoder
    pub fn max_past_prompt_tokens(&self) -> usize {
        self.whisper.max_past_prompt_tokens as usize
    }

    /// Indicates whether whisper is set to translate the output text to the specified output language
    pub fn translate(&self) -> bool {
        self.whisper.translate
    }

    /// Indicates the selected output language. None = auto = Auto-detect language.
    pub fn language(&self) -> &Option<Language> {
        &self.whisper.language
    }

    /// Indicates whether the gpu should be used.
    pub fn using_gpu(&self) -> bool {
        self.whisper.use_gpu
    }
    /// Indicates whether past context is used to prompt the decoder.
    pub fn using_no_context(&self) -> bool {
        self.whisper.use_no_context
    }
    /// Indicates whether flash attention is being used
    pub fn using_flash_attention(&self) -> bool {
        self.whisper.flash_attention
    }

    /// Gets a reference to the model being used for transcription
    pub fn model_id(&self) -> &Option<ModelId> {
        &self.whisper.model_id
    }

    // Realtime Accessors
    /// Gets the realtime timeout (in ms).
    pub fn realtime_timeout(&self) -> u128 {
        self.realtime.realtime_timeout
    }

    /// Gets the audio sampling window size (in ms).
    pub fn audio_sample_len_ms(&self) -> usize {
        self.realtime.audio_sample_len
    }

    /// Gets the voice-detection sampling window size (in ms).
    pub fn vad_sample_len(&self) -> usize {
        self.realtime.vad_sample_len
    }

    /// Gets the inner WhisperConfigsV2
    pub fn to_whisper_v2_configs(&self) -> &WhisperConfigsV2 {
        &self.whisper
    }

    /// Gets the inner RealtimeConfigs
    pub fn to_realtime_configs(&self) -> &RealtimeConfigs {
        &self.realtime
    }

    /// Consumes this object and converts to WhisperConfigsV2
    pub fn into_whisper_v2_configs(self) -> WhisperConfigsV2 {
        self.whisper
    }

    /// Consumes this object and converts to RealtimeConfigs
    pub fn into_realtime_configs(self) -> RealtimeConfigs {
        self.realtime
    }

    /// Constructs a FullParams object ready to be passed to [whisper_rs::WhisperState::full]
    /// Note: these configurations do not cover FullParams in entirety
    /// Features are exposed on an as-needed bases.
    /// See: [whisper_rs::FullParams] for documentation.
    pub fn to_whisper_full_params(&self) -> whisper_rs::FullParams {
        let mut params = self.whisper.to_whisper_full_params();
        // Forcing single segment transcription helps alleviate transcription artifacts when
        // running realtime to reduce the amount of false negatives in the
        // word-boundary resolution algorithm
        params.set_single_segment(true);
        params
    }
    /// Constructs a WhisperContextParameters object used to build [whisper_rs::WhisperContext]
    pub fn to_whisper_context_params(&self) -> whisper_rs::WhisperContextParameters {
        self.whisper.to_whisper_context_params()
    }
}

impl Default for WhisperRealtimeConfigs {
    fn default() -> Self {
        Self::new()
            .with_whisper_configs(
                WhisperConfigsV2::default()
                    // Whisper has trouble if context is maintained for subsequent transcription when
                    // streaming in realtime (usually resulting in duplicated output hallucinations)
                    // so context remembering is removed by default
                    .set_use_no_context(true)
                    .set_gpu(true)
                    .set_flash_attention(true),
            )
            .with_realtime_configs(RealtimeConfigs::default())
    }
}

// Legacy Implementation
/// Legacy configurations implementation. This is deprecated and will eventually be removed.
/// Note: voice_probability_threshold has been moved out of configs and into the VAD API.
/// If this value needs to be preserved, access it publicly, cache it, and store/set appropriately
/// before consuming into WhisperConfigsV2 or WhisperRealtimeConfigs.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub struct WhisperConfigsV1 {
    pub n_threads: std::ffi::c_int,
    pub set_translate: bool,
    pub language: Option<String>,
    // This is more or less a necessity for realtime.
    pub use_gpu: bool,

    pub model: DefaultModelType,

    // in milliseconds.
    pub realtime_timeout: u128,
    pub audio_sample_ms: usize,
    pub vad_sample_ms: usize,
    pub phrase_timeout: usize,

    pub voice_probability_threshold: f32,
    // To determine high-pass
    pub naive_vad_freq_threshold: f64,
    // To determine VAD for naive impl.
    pub naive_vad_energy_threshold: f64,
    pub naive_window_len: f64,
    pub naive_window_step: f64,
    // Stdout only.
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
}

impl WhisperConfigsV1 {
    /// Consumes and converts into WhisperConfigsV2.
    /// Since the new model implementation assumes some sort of id-value mapping infrastructure
    /// will be used, a ModelId hash is computed to store in the configs for quick data retrieval.
    /// See: [crate::whisper::model::ModelBank] and [crate::whisper::model::DefaultModelBank]
    pub fn into_v2(self) -> WhisperConfigsV2 {
        let model = self.model.to_model();
        let mut hasher = DefaultHasher::new();
        self.into_v2_with_model(model, &mut hasher)
    }

    /// Consumes and converts into WhisperConfigsV2 with the given hasher.
    /// Since the new model implementation assumes some sort of id-value mapping infrastructure
    /// will be used, a ModelId hash is computed to store in the configs for quick data retrieval.
    /// See: [crate::whisper::model::ModelBank] and [crate::whisper::model::DefaultModelBank]
    pub fn into_v2_with_hasher(self, hasher: &mut impl Hasher) -> WhisperConfigsV2 {
        let model = self.model.to_model();
        self.into_v2_with_model(model, hasher)
    }

    /// Consumes and converts into WhisperRealtimeConfigs, uses a default hasher to compute the
    /// ModelId.
    pub fn into_realtime_v1(self) -> WhisperRealtimeConfigs {
        let realtime_configs = self.to_realtime_configs();
        let whisper_configs = self.into_v2();
        WhisperRealtimeConfigs::new()
            .with_realtime_configs(realtime_configs)
            .with_whisper_configs(whisper_configs)
    }

    /// Consumes and converts into WhisperRealtimeConfigs using the provided hasher to compute the
    /// ModelId
    pub fn into_realtime_v1_with_hasher<H: Hasher>(self, hasher: &mut H) -> WhisperRealtimeConfigs {
        let realtime_configs = self.to_realtime_configs();
        let whisper_configs = self.into_v2_with_hasher(hasher);
        WhisperRealtimeConfigs::new()
            .with_realtime_configs(realtime_configs)
            .with_whisper_configs(whisper_configs)
    }

    fn into_v2_with_model<H: Hasher>(self, model: Model, hasher: &mut H) -> WhisperConfigsV2 {
        let language = self
            .language
            .as_ref()
            .map(|lang| Language::from_str(lang).unwrap());
        model.file_name().hash(hasher);
        let model_id = hasher.finish();
        WhisperConfigsV2::default()
            .with_n_threads(self.n_threads as usize)
            .set_translate(self.set_translate)
            .with_language(language)
            .set_gpu(self.use_gpu)
            .with_model_id(Some(model_id))
    }

    // Extracts the realtime-related information from WhisperConfigsV1 without consuming self,
    // so that self may be fully consumed into WhisperConfigsV2 (with or without a model prefix path)
    fn to_realtime_configs(&self) -> RealtimeConfigs {
        RealtimeConfigs::new()
            .with_audio_sample_len(self.audio_sample_ms)
            .with_vad_sample_len(self.vad_sample_ms)
            .with_realtime_timeout(self.realtime_timeout)
    }
}

pub const MAX_PROMPT_TOKENS: usize = 16384;
// Recommended 1Hr.
pub const REALTIME_AUDIO_TIMEOUT: u128 = std::time::Duration::new(3600, 0).as_millis();
pub const VAD_SAMPLE_MS: usize = 300;
// in ms
pub const AUDIO_SAMPLE_MS: usize = 10000;
