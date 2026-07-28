def risk_label(prediction):
    labels = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }

    return labels.get(prediction, "Unknown")