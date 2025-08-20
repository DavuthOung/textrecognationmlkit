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
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.text_recognation_ml_kit.ui.theme.ScanTheme
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults


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
private fun CropCard(crop: CroppedDetection) {
    Card(
        modifier = Modifier
            .width(180.dp)
            .height(180.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Box(Modifier.fillMaxSize()) {
            Image(
                bitmap = crop.bitmap.asImageBitmap(),
                contentDescription = "${crop.label.text} crop",
                modifier = Modifier.fillMaxSize()
            )
            // label ribbon
            Box(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(6.dp)
                    .background(
                        color = Color(0x88000000),
                        shape = RoundedCornerShape(6.dp)
                    )
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                Text(
                    text = "${crop.label.text}  ${(crop.label.confidence * 100f).toInt()}%",
                    color = Color.White,
                    style = MaterialTheme.typography.labelMedium
                )
            }
        }
    }
}


@Composable
fun ScanScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current

    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var imageBitmap by remember { mutableStateOf<Bitmap?>(null) }

    var crops: CroppedDetection

    // We’ll keep your state shape but use our own Label class
    val detectedObjectsInfo by remember {
        mutableStateOf<List<Pair<Rect, List<YoloLabel>>>>(emptyList())
    }

    // Create the YOLO interpreter once
    val yolo by remember {
        mutableStateOf(
            YoloTFLite(
                context = context,
                modelPath = "models/mrz_best_float32.tflite",
                scoreThreshold = 0.60f,          // <-- as you requested
                iouThreshold = 0.50f
            )
        )
    }

    DisposableEffect(Unit) {
        onDispose { yolo.close() }
    }

    fun runYolo(bitmap: Bitmap) {
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

            // Or: crop and resize each to 320x320 with letterbox (no distortion)
            val crops320 = cropDetections(
                src = bitmap,
                detections = dets,
                padPx = 4,
                padRatio = 0.04f,
                clampToImage = true,
                targetWidth = 320,
                targetHeight = 320,
                letterbox = true
            )

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
    fun loadAndProcessSampleImage() {
        try {
            val sampleBitmap = BitmapFactory.decodeResource(context.resources, R.drawable.idcard1)
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
        Button(onClick = { loadAndProcessSampleImage() }) {
            Text("Detect Objects in Sample Image")
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
                            topLeft = androidx.compose.ui.geometry.Offset(scaledLeft, scaledTop),
                            size = androidx.compose.ui.geometry.Size(
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
                items(count = crops.size, key = { idx -> "${crops[idx].label.index}-$idx" }) { idx ->
                    CropCard(crop = crops[idx])
                }
            }
        }

        if (detectedObjectsInfo.isNotEmpty()) {
            Text("Detected Objects:", style = MaterialTheme.typography.headlineSmall)
            detectedObjectsInfo.forEach { (box, labels) ->
                Text("Bounding Box: $box")
                labels.forEach { label ->
                    Text("  - Label: ${label.text}, Confidence: ${label.confidence.toStringWithDecimals(2)}")
                }
                Spacer(modifier = Modifier.height(8.dp))
            }
        } else if (!isLoading && errorMessage == null) {
            Text("No objects detected or no image processed yet.")
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
