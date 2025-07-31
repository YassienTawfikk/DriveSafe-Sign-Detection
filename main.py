import pandas as pd
from pathlib import Path
from src.__01__data_setup import *
from src.__02__CNN_model_training import *
from src.__03__CNN_model_evaluation import save_report_confusion_matrix


def main():
    # Step 1: Prepare data
    print("🔽 Downloading GTSRB from KaggleHub...")
    download_dataset()

    print("🔽 Plotting class distribution...")
    save_class_distribution()

    print("🔽 Load Preprocessed Training Data...")
    train_data, train_labels = load_preprocessed_data("Train.csv")

    print("🔽 Split Training Data into Train/Validation Sets...")
    x_train, x_val, y_train, y_val = train_validation_split(train_data, train_labels)

    print("🔽 Calling CNN Model...")
    model = return_CNN_model()

    print("🔽 Performing Training (8~10 Minutes)...")
    history = model.fit(x_train, y_train, batch_size=128, epochs=35, validation_data=(x_val, y_val))
    model.save('outputs' / 'model' / 'custom_traffic_classifier.h5')

    print("🔽 Save Training Performance...")
    save_training_performance(history=history)

    print("🔽 Load Preprocessed Test Data...")
    test_data, test_labels = load_preprocessed_data("Test.csv")

    print("🔽 Predicting Test Data...")
    pred_probs = model.predict(test_data)
    y_pred = np.argmax(pred_probs, axis=1)

    print("🔽 Save Model Evaluation Report in outputs/docs...")
    print("🔽 Save Confusion Matrix in outputs/figures...")
    save_report_confusion_matrix(test_labels, y_pred)


if __name__ == "__main__":
    main()
