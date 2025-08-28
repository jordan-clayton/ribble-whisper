// Model unit tests: Checking model path coherence, DefaultModelType url canonicalization
#[cfg(test)]
mod model_tests {
    use ribble_whisper::whisper::model::DefaultModelType;

    // TODO: implement ModelBank tests.
    #[test]
    fn test_url() {
        let model_type = DefaultModelType::default();

        let test_url = model_type.url();
        let test_url_str = test_url.as_str();
        const EXPECTED_URL_STR: &str =
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin";

        assert_eq!(
            test_url_str, EXPECTED_URL_STR,
            "Url malformed. Test: {}, Expected: {}",
            test_url_str, EXPECTED_URL_STR
        )
    }
    #[test]
    fn test_default_modeltype_name() {
        let expected = "TinyEn";
        let model_type = DefaultModelType::default();
        // This gets called in "to_model()" to set the User-facing name of the model.
        let actual = model_type.as_ref();
        assert_eq!(
            actual, expected,
            "Name failure. Actual: {}, Expected {}",
            actual, expected
        )
    }
    #[test]
    fn test_default_modeltype_file_name() {
        let expected = "ggml-tiny.en.bin";
        let model_type = DefaultModelType::default();
        // This gets called in "to_model()" to set the User-facing name of the model.
        let actual = model_type.to_file_name();
        assert_eq!(
            actual, expected,
            "Name failure. Actual: {}, Expected {}",
            actual, expected
        )
    }
}
