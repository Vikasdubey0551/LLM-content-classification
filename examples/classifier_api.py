from content_classification.classifier import APIClassifier

classification_config = {
    "model_path": "model/yarn-mistral-7b-128k.Q4_K_M.gguf",
    "IAB_categories": True,
    "Gender": True,
    "Topics": True,
    "struct": {
        "IAB_categories": ["IAB1", "IAB2", "IAB3"],
        "Age_groups": ["Adult", "Teen"],
        "Topics": ["Topic1", "Topic2"],
        "Gender": ["Male"],
    },
}


if __name__ == "__main__":

    content = """Villa living embodies a unique blend of luxury, comfort, and tranquility, offering a haven 
    away from the hustle and bustle of city life. Nestled amidst serene landscapes or perched on scenic coastal cliffs, 
    villas beckon those seeking a retreat where relaxation and rejuvenation are paramount"""

    classifier = APIClassifier(classification_config)
    response = classifier.get_response(content)
    print(response)
