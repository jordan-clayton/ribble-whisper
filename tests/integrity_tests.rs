// Since this is not considered to be within the scope of the core library, these will only be addressed
// should there be a need.
// Roadmap:
// - implement proper unit testing using mocked directories to avoid relying on integration tests
//   and encountering filesystem instability.
// - fuzz-testing the regex.
mod common;
#[cfg(test)]
#[cfg(feature = "integrity")]
mod model_integrity_tests {
    // NOTE: These tests look a little gnarly due to a late refactoring decision that introduces a
    // "Model Bank" abstraction. See: [crate::whisper::model:::ModelBank] for details
    use std::collections::HashMap;

    use crate::common::prep_model_bank;
    use ribble_whisper::whisper::integrity_utils::CHECKSUM_RE;
    use ribble_whisper::whisper::model::{
        Checksum, DefaultModelType, ModelBank, ModelLocation, ModelRetriever,
    };

    fn delete_model(file_path: &std::path::Path) -> std::io::Result<()> {
        std::fs::remove_file(file_path)
    }

    #[test]
    fn test_regex() {
        let mut checksums = HashMap::new();

        let test_line =
            "| Model               | Disk    | SHA                                        |
| ------------------- | ------- | ------------------------------------------ |
| tiny                | 75 MiB  | `bd577a113a864445d4c299885e0cb97d4ba92b5f` |
| tiny-q5_1           | 31 MiB  | `2827a03e495b1ed3048ef28a6a4620537db4ee51` |
| tiny.en             | 75 MiB  | `c78c86eb1a8faa21b369bcd33207cc90d64ae9df` |
| large-v1            | 2.9 GiB | `b1caaf735c4cc1429223d5a74f0f4d0b9b59a299` |";
        for line in test_line.lines() {
            if let Some(c) = CHECKSUM_RE.captures(line) {
                let model = c[1].to_string();
                let sha = c[2].to_string();
                checksums.insert(model, sha);
            }
        }

        assert!(
            !checksums.is_empty(),
            "Zero length: Regex failed to parse text"
        );
        assert!(
            checksums.contains_key("tiny"),
            "Tiny Key missing: Regex failed to parse text"
        );
        assert!(
            checksums.contains_key("tiny-q5_1"),
            "Tiny q5 Key missing: Regex failed to parse text"
        );

        let actual_tiny = checksums.get("tiny").unwrap();
        let expected_tiny = "bd577a113a864445d4c299885e0cb97d4ba92b5f";
        let actual_tinyq5 = checksums.get("tiny-q5_1").unwrap();
        let expected_tinyq5 = "2827a03e495b1ed3048ef28a6a4620537db4ee51";
        let actual_tiny_en = checksums.get("tiny.en").unwrap();
        let expected_tiny_en = "c78c86eb1a8faa21b369bcd33207cc90d64ae9df";
        let actual_large_v1 = checksums.get("large-v1").unwrap();
        let expected_large_v1 = "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299";

        assert_eq!(
            actual_tiny, expected_tiny,
            "Value failure, tiny. Test: {}, Expected: {}",
            actual_tiny, expected_tiny
        );
        assert_eq!(
            actual_tinyq5, expected_tinyq5,
            "Value failure, tiny-q5_1. Test: {}, Expected: {}",
            actual_tinyq5, expected_tinyq5
        );

        assert_eq!(
            actual_tiny_en, expected_tiny_en,
            "Value failure, tiny.en. Test: {}, Expected: {}",
            actual_tiny_en, expected_tiny_en
        );

        assert_eq!(
            actual_large_v1, expected_large_v1,
            "Value failure, large-v1. Test: {}, Expected: {}",
            actual_large_v1, expected_large_v1,
        );
    }
    #[test]
    fn test_integrity_check_not_downloaded() {
        // LargeV1 is not included in the default implementation and needs to be added to the model bank, or prep_model_bank will crash.
        let model_type = DefaultModelType::LargeV1;
        let init_model_type = DefaultModelType::Medium;

        let (mut model_bank, _) = prep_model_bank(init_model_type);
        let model_id = model_bank.insert_model(model_type.to_model());
        assert!(
            model_id.is_ok(),
            "Failed to insert large model entry in DefaultModelBank"
        );
        let model_id = model_id.unwrap();

        let exists = model_bank.model_exists_in_storage(model_id);
        assert!(
            exists.is_ok(),
            "File IO in integrity test with large model."
        );

        if exists.unwrap() {
            let file_location = model_bank.retrieve_model(model_id);
            assert!(
                file_location.is_some(),
                "Model bank not returning file path."
            );
            let file_path = match file_location.unwrap() {
                ModelLocation::DynamicFilePath(path) => path,
                _ => unreachable!("Default model bank only uses DynamicFilePath"),
            };

            let deleted = delete_model(file_path.as_path());
            assert!(
                deleted.is_ok(),
                "Failed to delete model: {}",
                deleted.unwrap_err()
            )
        }

        let exists = model_bank.model_exists_in_storage(model_id);
        assert!(
            exists.is_ok(),
            "File IO in integrity test with large model."
        );

        assert!(!exists.unwrap(), "Model still exists in the directory");

        // Get the checksum.
        let checksum = model_type.get_checksum(model_bank.model_directory(), None);
        assert!(checksum.is_ok(), "{}", checksum.unwrap_err());
        let checksum = checksum.unwrap();
        let c = Checksum::Sha1(checksum.as_str());
        let verified = model_bank.verify_checksum(model_id, &c);
        assert!(verified.is_ok(), "File error: {}", verified.unwrap_err());

        assert!(
            !verified.unwrap(),
            "Checksum test passed despite file not existing. Test checksum: {}",
            checksum
        )
    }
    #[test]
    fn test_integrity_check_downloaded() {
        // NOTE: THIS TEST WILL FAIL IF THE MODEL ISN'T DOWNLOADED
        let model_type = DefaultModelType::Small;
        let (mut model_bank, model_id) = prep_model_bank(model_type);

        let exists = model_bank.model_exists_in_storage(model_id);
        assert!(
            exists.is_ok(),
            "File IO in integrity test with default model."
        );

        assert!(
            exists.unwrap(),
            "Failed to find default model in data directory"
        );

        // Get the checksum.
        let checksum = model_type.get_checksum(model_bank.model_directory(), None);
        assert!(checksum.is_ok(), "{}", checksum.unwrap_err());
        let checksum = checksum.unwrap();
        let c = Checksum::Sha1(checksum.as_str());

        let verified = model_bank.verify_checksum(model_id, &c);
        assert!(verified.is_ok(), "File error: {}", verified.unwrap_err());

        assert!(
            verified.unwrap(),
            "Checksum test failed despite file existing. Test checksum: {}",
            checksum
        )
    }
}
