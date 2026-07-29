def risk_label(prediction):
    labels = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }

    return labels.get(prediction, "Unknown")


def risk_color(prediction):
    colors = {
        0: "🟢",
        1: "🟡",
        2: "🔴"
    }

    return colors.get(prediction, "⚪")


def risk_recommendation(prediction):
    recommendations = {
        0: "Maintain current flood prevention measures.",
        1: "Monitor conditions and improve preparedness plans.",
        2: "Take immediate mitigation and emergency preparedness actions."
    }

    return recommendations.get(prediction, "No recommendation available.")