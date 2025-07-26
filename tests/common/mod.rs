use ribble_whisper::whisper::model::{
    DefaultModelBank, DefaultModelType, ModelBank, ModelId, ModelLocation, ModelRetriever,
};

// NOTE: this is not actually dead code. It's just a bit tricky to share common functions across
// test modules. I'm not quite sure if this is the "correct" way to do this, but it works nonetheless.
#[allow(dead_code)]
pub(crate) fn prep_model_bank(model_type: DefaultModelType) -> (DefaultModelBank, ModelId) {
    let bank = DefaultModelBank::new();

    let model_id = bank.get_model_id(model_type);
    // Checking -> These models -should- exist before they're being used in testing.
    let model = bank.get_model(model_id);

    assert!(
        model.is_some(),
        "Failed to retrieve model: {} from default bank.",
        model_type
    );

    let model = model.unwrap();
    let manual_join = bank.model_directory().join(model.file_name());
    let retrieve_join = bank.retrieve_model(model_id);
    assert!(
        retrieve_join.is_some(),
        "{} Model is not in bank.",
        model_type
    );
    let retrieve_join = match retrieve_join.unwrap() {
        ModelLocation::DynamicFilePath(path) => path,
        _ => unreachable!("DefaultModelBank only returns DynamicFilePath."),
    };

    assert_eq!(
        manual_join, retrieve_join,
        "Path error. Manual: {:?}, Retrieved: {:?}",
        manual_join, retrieve_join
    );

    (bank, model_id)
}
