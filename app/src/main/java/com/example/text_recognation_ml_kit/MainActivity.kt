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
import com.example.text_recognation_ml_kit.ui.theme.ScanTheme
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await


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

suspend fun ocrAllCrops(crops: List<CroppedDetection>): String? {

    val recognizer: TextRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    var results: String?

    return try {
        val c = crops[0]
        val image = InputImage.fromBitmap(c.bitmap, 0)
        val vt: Text = recognizer.process(image).await()
        results = vt.text.trim()
        results
    } finally {
        recognizer.close()
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

fun normalizeMRZText(rawText: String): String {
    Log.i("Raw", rawText)
    var normalizedText = ""
    for (char in rawText) {
        if(!char.isWhitespace()) {
            normalizedText += char.uppercase()
        }
    }
    return normalizedText
}

@Composable
fun ScanScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var imageBitmap by remember { mutableStateOf<Bitmap?>(null) }

    var crops by remember { mutableStateOf<List<CroppedDetection>>(emptyList()) }

    val sampleBitmap = BitmapFactory.decodeResource(context.resources, R.drawable.id_card)

    // We’ll keep your state shape but use our own Label class
    var detectedObjectsInfo by remember {
        mutableStateOf<List<Pair<Rect, List<YoloLabel>>>>(emptyList())
    }

    // Create the YOLO interpreter once
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

    suspend fun runYolo(bitmap: Bitmap) {
        isLoading = true
        errorMessage = null
        detectedObjectsInfo = emptyList()
        imageBitmap = bitmap

        try {
            val dets: List<YoloDetection> = yolo.detect(bitmap)
            // Crop with +6% padding and clamp to image, keep original size
            crops = cropDetections(
                src = bitmap,
                detections = dets,
                padPx = 0,
                padRatio = 0.06f,       // add ~6% around each side
                clampToImage = true
            )

            if(crops.isNotEmpty()){
                val result =  ocrAllCrops(crops)
                Log.i("CROP", "runYolo: \n${normalizeMRZText(result as String)}")
            } else {
                Log.i("CROP", "runYolo: no crops")
            }


            // Convert to Compose Rect (original image cords)
            val processed = dets.map { d ->
                val box = Rect(
                    left = d.box.left,
                    top = d.box.top,
                    right = d.box.right,
                    bottom = d.box.bottom
                )
                box to listOf(d.label)
            }
            detectedObjectsInfo = processed
            isLoading = false

            if (dets.isEmpty()) {
                Log.i(TAG, "No objects detected.")
            } else {
                dets.forEachIndexed { idx, d ->
                    Log.i(TAG, "Detection #$idx -> ${d.label.text} (${d.label.confidence.toStringWithDecimals(2)})")
                    Log.i(TAG, "Rect: L=${d.box.left.toInt()} T=${d.box.top.toInt()} R=${d.box.right.toInt()} B=${d.box.bottom.toInt()}")
                }
            }
        } catch (e: Exception) {
            isLoading = false
            errorMessage = "Detection failed: ${e.localizedMessage}"
            Log.e(TAG, "YOLO detection error", e)
        }
    }

    @SuppressLint("LocalContextResourcesRead")
    suspend fun loadAndProcessSampleImage() {
        try {
            runYolo(sampleBitmap)
        } catch (e: Exception) {
            errorMessage = "Error loading sample image: ${e.localizedMessage}"
            Log.e(TAG, "Error loading sample image", e)
        }
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

                    detectedObjectsInfo.forEach { (box) ->
                        val scaledLeft = box.left * scaleX
                        val scaledTop = box.top * scaleY
                        val scaledRight = box.right * scaleX
                        val scaledBottom = box.bottom * scaleY

                        drawRect(
                            color = Color.Red,
                            topLeft = Offset(scaledLeft, scaledTop),
                            size = Size(
                                scaledRight - scaledLeft,
                                scaledBottom - scaledTop
                            ),
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
                    CropCardWithText(crop = crop)
                }
            }
        }

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
