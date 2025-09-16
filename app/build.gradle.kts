plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.example.grapevineguardian"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.grapevineguardian"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    // The TensorFlow Lite model is obtained at runtime, so we intentionally avoid
    // packaging it as an Android asset in this module.
}

dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
}
