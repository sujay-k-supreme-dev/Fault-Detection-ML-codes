# Convert TFLite model to C header file for ESP32
tflite_model_path = "bearing_fault_model.tflite"
header_file_path = "teatingmodel.h"

with open(tflite_model_path, "rb") as f:
    tflite_model = f.read()

# Convert to byte array format
c_array = ", ".join(str(b) for b in tflite_model)

# Create C header file
with open(header_file_path, "w") as f:
    f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
    f.write("const unsigned char model_data[] = {" + c_array + "};\n")
    f.write("const unsigned int model_data_len = " + str(len(tflite_model)) + ";\n")
    f.write("\n#endif // MODEL_H")

print("✅ Model converted to C header file: model.h")
