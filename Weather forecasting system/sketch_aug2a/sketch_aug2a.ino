#include <SDHCI.h>
#include <SD.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

const char *model_path = "/mnt/sd0/weather_model.tflite";
const int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output_summary;
TfLiteTensor* output_precip_type;
TfLiteTensor* output_temperature;
TfLiteTensor* output_daily_summary;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!SD.begin(SDHCI)) {
    Serial.println("Failed to initialize SD card");
    return;
  }

  File modelFile = SD.open(model_path);
  if (!modelFile) {
    Serial.println("Failed to open model file");
    return;
  }

  size_t modelSize = modelFile.size();
  uint8_t* modelData = (uint8_t*)malloc(modelSize);
  modelFile.read(modelData, modelSize);
  modelFile.close();

  model = tflite::GetModel(modelData);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version is not supported");
    return;
  }

  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output_summary = interpreter->output(0);
  output_precip_type = interpreter->output(1);
  output_temperature = interpreter->output(2);
  output_daily_summary = interpreter->output(3);

  Serial.println("Setup complete. Enter parameters:");
  Serial.println("Format: temperature humidity wind_speed wind_bearing visibility pressure year month day hour");
  Serial.println("Example: 20.0 0.8 5.0 180 10.0 1013 2024 7 31 12");
}

void loop() {
  if (Serial.available() > 0) {
    String inputString = Serial.readStringUntil('\n');
    float features[10];
    sscanf(inputString.c_str(), "%f %f %f %f %f %f %f %f %f %f",
           &features[0], &features[1], &features[2], &features[3], &features[4],
           &features[5], &features[6], &features[7], &features[8], &features[9]);

    for (size_t i = 0; i < 10; i++) {
      input->data.f[i] = features[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Failed to invoke interpreter");
      return;
    }

    float predicted_summary = output_summary->data.f[0];
    float predicted_precip_type = output_precip_type->data.f[0];
    float predicted_temperature = output_temperature->data.f[0];
    float predicted_daily_summary = output_daily_summary->data.f[0];

    Serial.print("Predicted Summary: ");
    Serial.println(predicted_summary);
    Serial.print("Predicted Precip Type: ");
    Serial.println(predicted_precip_type);
    Serial.print("Predicted Temperature: ");
    Serial.print(predicted_temperature);
    Serial.println(" °C");
    Serial.print("Predicted Daily Summary: ");
    Serial.println(predicted_daily_summary);
  }
}

