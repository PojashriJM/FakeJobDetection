
import json
import zipfile
import os
from tensorflow.keras.models import load_model
import numpy as np

source = os.path.join(
    "backend", "flask_api", "models", "lstm.keras"
)

output = os.path.join(
    "backend", "flask_api", "models", "lstm_compatible.keras"
)

# Read the original model archive and patch its configuration
with zipfile.ZipFile(source, "r") as original:
    config = json.loads(original.read("config.json"))

    def remove_unsupported_fields(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for value in obj.values():
                remove_unsupported_fields(value)
        elif isinstance(obj, list):
            for item in obj:
                remove_unsupported_fields(item)

    remove_unsupported_fields(config)

    # Write a separate model archive, preserving all other files
    with zipfile.ZipFile(output, "w") as patched:
        for item in original.infolist():
            data = original.read(item.filename)

            if item.filename == "config.json":
                data = json.dumps(config).encode("utf-8")

            patched.writestr(item, data)

print("Patched model saved:", output)

# Test loading
try:
    model = load_model(output, compile=False)

    print("SUCCESS: LSTM loaded!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    sample = np.zeros((1, 300), dtype=np.int32)
    prediction = model.predict(sample, verbose=0)

    print("Test prediction:", prediction)

except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:1000])
