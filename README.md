# forCodex

forCodex

## TensorFlow Lite model setup

The application no longer ships the `grapevine_disease.tflite` model inside the APK. Before
running a build you must provide the model file yourself so it can be referenced at runtime.

1. Obtain the `grapevine_disease.tflite` artifact from your preferred distribution channel
   (for example, the ML training pipeline output or an internal storage bucket). If you have
   shell access you can download it with `curl`:
   ```bash
   curl -L "https://example.com/path/to/grapevine_disease.tflite" -o grapevine_disease.tflite
   ```
2. Copy the file onto the device or emulator in a directory the app can read. A common pattern
   is to store it inside the app's files directory so it remains private:
   ```bash
   adb push grapevine_disease.tflite /sdcard/Android/data/com.example.grapevineguardian/files/models/
   ```
3. Update your app configuration (for example, via user settings or dependency injection) to
   point `DetectionEngine` at the absolute path of the copied model file before running
   inference. The engine validates that the supplied path exists and is readable.

With these steps complete the application will load the TensorFlow Lite model from the provided
location at startup and perform detections without bundling the asset in the APK.
