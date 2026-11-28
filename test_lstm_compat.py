
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, Dense


class CompatibleEmbedding(Embedding):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


class CompatibleDense(Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


model_path = os.path.join(
    "backend", "flask_api", "models", "lstm.keras"
)

try:
    model = load_model(
        model_path,
        compile=False,
        custom_objects={
            "Embedding": CompatibleEmbedding,
            "Dense": CompatibleDense
        }
    )

    print("LSTM loaded successfully!")
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    # Test inference using a dummy sequence
    sample = np.zeros((1, 300), dtype=np.int32)
    prediction = model.predict(sample, verbose=0)

    print("Test prediction:", prediction)

except Exception as e:
    print("LSTM loading failed:", type(e).__name__, str(e))
