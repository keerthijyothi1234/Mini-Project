def predict_cyberattack(request, data):
    model = load_pru_model()  # Load PRU Model
    prediction = model.predict(data)
    return prediction