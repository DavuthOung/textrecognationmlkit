package com.example.text_recognation_ml_kit

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.text_recognation_ml_kit.ui.theme.ScanTheme
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc
import androidx.core.graphics.createBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.core.Mat

const val TAG = "ObjectDetectionDemo"
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContent {
            ScanTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ScanScreen()
                }
            }
        }
    }
}

@Composable
private fun CropCardWithText(crop: CroppedDetection) {
    val maxW = 180.dp
    val maxH = 180.dp
    val density = LocalDensity.current

    // Use the actual bitmap size (recommended)
    val bmpW = crop.bitmap.width
    val bmpH = crop.bitmap.height

    // Compute target dp size that fits within maxW×maxH while preserving aspect ratio
    val (targetW, targetH) = remember(bmpW, bmpH, maxW, maxH) {
        with(density) {
            val maxWPx = maxW.toPx()
            val maxHPx = maxH.toPx()
            val scale = minOf(maxWPx / bmpW, maxHPx / bmpH)
            (bmpW * scale).toDp() to (bmpH * scale).toDp()
        }
    }

    Card(elevation = CardDefaults.cardElevation(4.dp)) {
        Column(Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Image(
                bitmap = crop.bitmap.asImageBitmap(),
                contentDescription = "${crop.label.text} crop",
                modifier = Modifier.size(targetW, targetH),
                contentScale = ContentScale.Fit
            )
        }
    }
}

@SuppressLint("LocalContextResourcesRead")
@Composable
fun ScanScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var imageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    val tesseract = Tesseract(context) // Assuming Tesseract class is correctly implemented
    var crops by remember { mutableStateOf<List<CroppedDetection>>(emptyList()) }
    val sampleBitmap = BitmapFactory.decodeResource(context.resources, R.drawable.id_card)

    // --- SOLUTION: Declare uiStateVariable as Compose State ---
    var uiStateVariable by remember { mutableStateOf<String?>("Tap 'Detect MRZ' to start") }

    var detectedObjectsInfo by remember {
        mutableStateOf<List<Pair<Rect, List<YoloLabel>>>>(emptyList())
    }

    suspend fun extractTextWithTesseract(tesseract: Tesseract, bitmap: Bitmap): String =
        withContext(Dispatchers.Default) {
            tesseract.extractText(bitmap) as String // Ensure Tesseract's extractText is safe to call from background
        }

    val yolo by remember {
        mutableStateOf(
            YoloTFLite(
                context = context,
                modelPath = "models/mrz_best_float32.tflite",
                scoreThreshold = 0.60f,
                iouThreshold = 0.50f
            )
        )
    }

    DisposableEffect(Unit) {
        onDispose { yolo.close() }
    }

    suspend fun runYolo(bitmapToProcess: Bitmap) { // Renamed parameter to avoid confusion with state
        withContext(Dispatchers.Main) {
            isLoading = true
            errorMessage = null
            imageBitmap = bitmapToProcess // Update the displayed image
            detectedObjectsInfo = emptyList()
            uiStateVariable = "Processing..." // Initial message during processing
        }

        try {
            val dets: List<YoloDetection> = withContext(Dispatchers.Default) {
                yolo.detect(bitmapToProcess)
            }

            val currentCrops = withContext(Dispatchers.Default) {
                cropDetections(
                    src = bitmapToProcess,
                    detections = dets,
                    padPx = 0,
                    padRatio = 0.06f,
                    clampToImage = true
                )
            }
            // Update the crops state on the main thread
            withContext(Dispatchers.Main) {
                crops = currentCrops
            }


            if (currentCrops.isNotEmpty()) {
                if (OpenCVLoader.initLocal()) { // Or any other OpenCV init method
                    Log.i(TAG, "OpenCV loaded successfully")

                    val firstCropOriginalBitmap = currentCrops[0].bitmap

                    val inputMat = Mat()
                    withContext(Dispatchers.Default) {
                        Utils.bitmapToMat(firstCropOriginalBitmap, inputMat)
                        Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGBA2BGR)
                    }

                    val preprocessedMat = preprocessForOcrAdvanced( // This is a suspend function
                        inputBgr = inputMat,
                        targetDpi = 300,
                        estimatedSourceDpi = 150,
                        denoiseStrength = 8f,
                        binarizationMethod = "ADAPTIVE_GAUSSIAN",
                        adaptiveBlockSize = 15,
                        adaptiveC = 4.0,
                        attemptDeskew = true
                    )
                    inputMat.release()

                    if (!preprocessedMat.empty()) {
                        val preprocessedBitmapForTesseract = withContext(Dispatchers.Default) {
                            val tempBitmap = createBitmap(
                                preprocessedMat.cols(),
                                preprocessedMat.rows(),
                                Bitmap.Config.ARGB_8888
                            )
                            val tempRgbaMat = Mat()
                            Imgproc.cvtColor(preprocessedMat, tempRgbaMat, Imgproc.COLOR_BGR2RGBA)
                            Utils.matToBitmap(tempRgbaMat, tempBitmap)
                            tempRgbaMat.release()
                            tempBitmap // Return the bitmap
                        }
                        preprocessedMat.release()

                        val mrzText = extractTextWithTesseract(tesseract, preprocessedBitmapForTesseract)
                        val parsedInfo = MrzParser.parse(mrzText)
                        withContext(Dispatchers.Main) {
                            Log.i("MRZ_PREPROCESSED", mrzText)
                            uiStateVariable = mrzText // This will now trigger recomposition
                            if (parsedInfo.parsingErrors.isNotEmpty()) {
                                Log.e("MRZ_APP", "Parsing Errors: ${parsedInfo.parsingErrors.joinToString("\n")}")
                                // Update UI to show "Could not parse MRZ" or specific errors
                            } else {
                                Log.i("MRZ_APP", "Parsed Successfully: $parsedInfo")
                                // Update your UI state variables with fields from parsedInfo
                                // e.g., uiSurnameState = parsedInfo.surname
                                //       uiDobState = formatMyDate(parsedInfo.dateOfBirth) // you'll need a date formatter
                                //       uiOverallValidityState = parsedInfo.overallValid.toString()
                            }
                        }
                    } else {
                        Log.w(TAG, "Preprocessing resulted in an empty Mat.")
                        withContext(Dispatchers.Main) {
                            uiStateVariable = "Preprocessing failed (empty result)"
                        }
                    }
                } else {
                    Log.e(TAG, "OpenCV initialization failed!")
                    withContext(Dispatchers.Main) {
                        uiStateVariable = "OpenCV init failed"
                    }
                }
            } else {
                withContext(Dispatchers.Main) {
                    uiStateVariable = "No MRZ detected to process."
                }
            }

            withContext(Dispatchers.Main) {
                val processedInfo = dets.map { d ->
                    val box = Rect(d.box.left, d.box.top, d.box.right, d.box.bottom)
                    box to listOf(d.label)
                }
                detectedObjectsInfo = processedInfo
                isLoading = false

                if (dets.isEmpty()) Log.i(TAG, "No objects detected.")
                else dets.forEachIndexed { idx, d ->
                    Log.i(
                        TAG,
                        "Detection #$idx -> ${d.label.text} (${
                            d.label.confidence.toStringWithDecimals(2)
                        })"
                    )
                }
                if (uiStateVariable == "Processing...") { // If OCR didn't update it
                    uiStateVariable =
                        if (dets.isNotEmpty() && currentCrops.isEmpty()) "Objects detected, but no MRZ crop found." else "Detection complete."
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Processing error in runYolo", e)
            withContext(Dispatchers.Main) {
                isLoading = false
                errorMessage = "Processing failed: ${e.localizedMessage}"
                uiStateVariable = "Error: ${e.localizedMessage?.take(100)}"
            }
        }
    }


    @SuppressLint("LocalContextResourcesRead")
    suspend fun loadAndProcessSampleImage() {
        runYolo(sampleBitmap) // Pass the sampleBitmap to runYolo
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(top = 50.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Button(onClick = {
            coroutineScope.launch {
                loadAndProcessSampleImage()
            }
        }) {
            Text("Detect MRZ")
        }

        if (isLoading) CircularProgressIndicator()

        errorMessage?.let { Text(text = "Error: $it", color = MaterialTheme.colorScheme.error) }

        imageBitmap?.let { bmp ->
            Box {
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = "Processed Image",
                    modifier = Modifier.fillMaxWidth()
                )
                Canvas(modifier = Modifier.matchParentSize()) {
                    val scaleX = size.width / bmp.width.toFloat()
                    val scaleY = size.height / bmp.height.toFloat()

                    detectedObjectsInfo.forEach { (box, _) -> // Changed from (box) to (box, _)
                        val scaledLeft = box.left * scaleX
                        val scaledTop = box.top * scaleY
                        val scaledRight = box.right * scaleX
                        val scaledBottom = box.bottom * scaleY

                        drawRect(
                            color = Color.Red,
                            topLeft = Offset(scaledLeft, scaledTop),
                            size = Size(scaledRight - scaledLeft, scaledBottom - scaledTop),
                            style = Stroke(width = 2.dp.toPx())
                        )
                    }
                }
            }
        }

        if (crops.isNotEmpty()) {
            Text("Cropped Results:", style = MaterialTheme.typography.headlineSmall)
            Spacer(Modifier.height(8.dp))

            LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                items(crops.size) { idx ->
                    val crop = crops[idx]
                    CropCardWithText(crop = crop) // Ensure CropCardWithText is implemented
                }
            }
        }

        Text(
            text = "Detected Text: ${uiStateVariable ?: "N/A"}", // Provide a fallback if null
            style = MaterialTheme.typography.headlineSmall.copy(
                fontSize = 14.sp
            )
        )
    }
}

private fun Float.toStringWithDecimals(n: Int) = "%.${n}f".format(this)
@Preview(showBackground = true)
@Composable
fun TextRecognitionScreenPreview() {
    ScanTheme {
        ScanScreen()
    }
}
