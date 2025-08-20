package com.example.text_recognation_ml_kit.screens


import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions

// ...
// 1. Configure the local model source
val localModel = LocalModel.Builder()
    .setAssetFilePath("models/mrz_best_float32.tflite") // Correct path relative to assets
    .build()

