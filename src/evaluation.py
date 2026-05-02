import os
import pandas as pd
import joblib


def compare_and_save(dt_model, rf_model, dt_accuracy, rf_accuracy):

    # Compare models
    print("\n== Model Comparison ==")
    print(f"Decision Tree accuracy : {dt_accuracy:.4f}")
    print(f"Random Forest accuracy : {rf_accuracy:.4f}")

    if rf_accuracy > dt_accuracy:
        print("Best model: Random Forest")
    else:
        print("Best model: Decision Tree")

    print("_" * 50)

    # Save models
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(dt_model, 'outputs/models/decision_tree.joblib')
    joblib.dump(rf_model, 'outputs/models/random_forest.joblib')
    print("Models saved successfully.")

    # Save comparison table
    os.makedirs('outputs/tables', exist_ok=True)
    comparison = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest'],
        'Accuracy': [dt_accuracy, rf_accuracy]
    })
    comparison.to_csv('outputs/tables/model_comparison.csv', index=False)
    print("Comparison table saved successfully.")
