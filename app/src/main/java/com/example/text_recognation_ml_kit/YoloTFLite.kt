package com.example.text_recognation_ml_kit
import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import androidx.core.graphics.scale
import androidx.core.graphics.createBitmap
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.gpu.GpuDelegate
import kotlin.math.roundToInt

data class YoloLabel(val text: String, val confidence: Float, val index: Int)
data class YoloDetection(val box: RectF, val label: YoloLabel)

// Put near your other data classes
data class CroppedDetection(
    val bitmap: Bitmap,
    val rect: Rect,              // crop rect in ORIGINAL image coords (ints)
    val label: YoloLabel
)

/**
 * Crop all detected boxes from the original image.
 *
 * @param src          The ORIGINAL bitmap you ran detection on.
 * @param detections   Detections whose boxes are in original image coords.
 * @param padPx        Extra padding in pixels to add on each side (after padRatio).
 * @param padRatio     Extra padding as a fraction of the box size (e.g., 0.10 = +10% width/height).
 * @param clampToImage If true, crops are clamped so they never go outside src bounds.
 * @param targetWidth  Optional resize width. If set with targetHeight, the crop will be resized.
 * @param targetHeight Optional resize height. Use with targetWidth.
 * @param letterbox    If true and resizing, keep aspect ratio with padding (letterbox). If false, stretch.
 * @param bgColor      Letterbox background color if letterbox=true.
 */
fun cropDetections(
    src: Bitmap,
    detections: List<YoloDetection>,
    padPx: Int = 0,
    padRatio: Float = 0f,
    clampToImage: Boolean = true,
    targetWidth: Int? = null,
    targetHeight: Int? = null,
    letterbox: Boolean = false,
    bgColor: Int = Color.BLACK
): List<CroppedDetection> {
    val W = src.width
    val H = src.height
    val out = mutableListOf<CroppedDetection>()

    detections.forEachIndexed { i, det ->
        val b = RectF(det.box) // already in original image coords
        val w = b.width()
        val h = b.height()
        if (w <= 0f || h <= 0f) return@forEachIndexed

        // Expand by ratio (relative to box) + absolute pixels
        val dx = (w * padRatio) / 2f + padPx
        val dy = (h * padRatio) / 2f + padPx
        b.inset(-dx, -dy)

        // Clamp if requested
        val left   = if (clampToImage) b.left.coerceIn(0f, (W - 1).toFloat()) else b.left
        val top    = if (clampToImage) b.top.coerceIn(0f, (H - 1).toFloat()) else b.top
        val right  = if (clampToImage) b.right.coerceIn(1f, W.toFloat()) else b.right
        val bottom = if (clampToImage) b.bottom.coerceIn(1f, H.toFloat()) else b.bottom

        val l = left.toInt().coerceAtLeast(0)
        val t = top.toInt().coerceAtLeast(0)
        val r = right.toInt().coerceAtMost(W)
        val btm = bottom.toInt().coerceAtMost(H)

        val cw = (r - l).coerceAtLeast(1)
        val ch = (btm - t).coerceAtLeast(1)
        if (cw <= 0 || ch <= 0) return@forEachIndexed

        // Create the crop
        val crop = try {
            Bitmap.createBitmap(src, l, t, cw, ch)
        } catch (e: Exception) {
            Log.w(TAG, "cropDetections: skipped invalid crop #$i  rect=[$l,$t,$r,$btm]", e)
            return@forEachIndexed
        }

        val finalBmp: Bitmap = if (targetWidth != null && targetHeight != null) {
            if (!letterbox) {
                crop.scale(targetWidth, targetHeight)
            } else {
                // Letterbox into target size keeping aspect ratio
                val tw = targetWidth
                val th = targetHeight
                val scale = minOf(tw.toFloat() / cw, th.toFloat() / ch)
                val newW = (cw * scale).toInt().coerceAtLeast(1)
                val newH = (ch * scale).toInt().coerceAtLeast(1)
                val dx2 = (tw - newW) / 2f
                val dy2 = (th - newH) / 2f

                val canvasBmp = createBitmap(tw, th)
                val c = Canvas(canvasBmp)
                c.drawColor(bgColor)
                val scaled = crop.scale(newW, newH)
                c.drawBitmap(scaled, dx2, dy2, null)
                scaled.recycle()
                canvasBmp
            }
        } else crop

        // Log useful info
        Log.i(TAG, "CROP #$i  label=${det.label.text} conf=${"%.2f".format(det.label.confidence)}  " +
                "rect=[L=$l T=$t R=$r B=$btm]  out=${finalBmp.width}x${finalBmp.height}")

        out += CroppedDetection(
            bitmap = finalBmp,
            rect = Rect(l, t, r, btm),
            label = det.label
        )

        if (finalBmp !== crop) {
            // If we created a resized/letterboxed bitmap, free the intermediate crop to save memory
            crop.recycle()
        }
    }

    Log.i(TAG, "cropDetections: produced ${out.size} crops from ${detections.size} detections")
    return out
}


class YoloTFLite(
    private val context: Context,
    private val modelPath: String = "models/mrz_best_float32.tflite",
    private val scoreThreshold: Float = 0.60f,
    private val iouThreshold: Float = 0.50f,

    // === Pre/Post toggles (tune these if needed) ===
    private val normalizeTo01: Boolean = true,     // true → divide pixels by 255f
    private val channelsBGR: Boolean = false,      // true → feed B,G,R instead of R,G,B
    private val useLetterbox: Boolean = true,      // true → keep aspect ratio (recommended)
    private val coordsAreNormalized: Boolean = true, // true → cx,cy,w,h in 0..1 (your export likely yes)
    private val applySigmoidToScore: Boolean = false, // flip true if scores look raw/logits
    private val hasSeparateObjAndCls: Boolean = false // set true only if A==6 and your last 2 chans are obj+cls
) {
    companion object {
        private const val TAG = "YoloTFLite"
    }

    private val interpreter: Interpreter
    private val labels: List<String>
    private val inputWidth: Int
    private val inputHeight: Int

    init {
        interpreter = Interpreter(loadModelFile(context, modelPath), Interpreter.Options().apply {
             setNumThreads(4)
            // GPU optional (add tflite-gpu dependency first):
             addDelegate(GpuDelegate())
        })

        val inShape = interpreter.getInputTensor(0).shape() // [1, H, W, 3]
        inputHeight = inShape[1]
        inputWidth = inShape[2]

        labels = loadLabels(context, "models/labelmap.txt")

        // Log outputs once for visibility
        val outCount = interpreter.outputTensorCount
        Log.i(TAG, "TFLite outputs: count=$outCount")
        for (i in 0 until outCount) {
            val t = interpreter.getOutputTensor(i)
            Log.i(TAG, "  out#$i  shape=${t.shape().contentToString()}  type=${t.dataType()}")
        }
    }

    fun close() = interpreter.close()

    fun detect(src: Bitmap): List<YoloDetection> {
        // 1) Preprocess
        val pp = if (useLetterbox) letterbox(src, inputWidth, inputHeight, Color.BLACK)
        else PP(resize(src, inputWidth, inputHeight),  // stretch
            src.width.toFloat() / inputWidth,      // dummy inverses (we won’t use)
            0f, 0f)
        val input = bitmapToFloatBuffer(pp.letterboxed, normalizeTo01, channelsBGR)

        // 2) Inspect output 0
        val out0 = interpreter.getOutputTensor(0)
        val s = out0.shape() // expected [1, 5, N] for your case
        // We’ll branch on common shapes
        val results: List<YoloDetection> = when {
            s.size == 3 && s[0] == 1 && s[1] in 5..6 -> {
                // channels-first [1, A, N]
                val A = s[1];
                val N = s[2]
                val cf = Array(1) { Array(A) { FloatArray(N) } } // [1, A, N]
                interpreter.run(input, cf)
                decodeChannelsFirst5xN(cf[0], src, pp, A, N)
            }
            s.size == 3 && s[0] == 1 -> {
                // row-major [1, N, A]   (e.g., [1, 25200, 85])
                val N = s[1]; val A = s[2]
                val rm = Array(1) { Array(N) { FloatArray(A) } } // [1, N, A]
                interpreter.run(input, rm)
                decodeRowMajorNxA(rm[0], src, pp, N, A)
            }
            else -> {
                // Fallback: run multiple outputs if any (rare for your export)
                val outCount = interpreter.outputTensorCount
                val outputs = HashMap<Int, Any>(outCount)
                for (i in 0 until outCount) {
                    val sh = interpreter.getOutputTensor(i).shape()
                    outputs[i] = Array(sh[0]) { Array(sh[1]) { FloatArray(sh[2]) } }
                }
                interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
                // Concatenate all heads assuming [1,N_i,A] shape
                val rows = mutableListOf<FloatArray>()
                outputs.forEach { (_, anyArr) ->
                    val arr = anyArr as Array<Array<FloatArray>>
                    arr[0].forEach { rows += it }
                }
                decodeRowMajorNxA(rows.toTypedArray(), src, pp, rows.size, rows.firstOrNull()?.size ?: 0)
            }
        }

        // 3) NMS
        val kept = nms(results, iouThreshold)

        // 4) Pretty debug log
        debugReport(src, kept)

        return kept
    }

    // ------------------ Decoders ------------------

    // For channels-first outputs: [A, N] with A == 5 or 6
    private fun decodeChannelsFirst5xN(
        channels: Array<FloatArray>, // [A, N]
        src: Bitmap,
        pp: PP,
        A: Int,
        N: Int
    ): List<YoloDetection> {
        val out = mutableListOf<YoloDetection>()
        var maxScoreSeen = 0f
        val X = 0; val Y = 1; val W = 2; val H = 3

        for (j in 0 until N) {
            var cx = channels[X][j]
            var cy = channels[Y][j]
            var w  = channels[W][j]
            var h  = channels[H][j]

            val score = when {
                A >= 6 && hasSeparateObjAndCls -> {
                    var obj = channels[4][j]
                    var cls = channels[5][j]
                    if (applySigmoidToScore) { obj = sigmoid(obj); cls = sigmoid(cls) }
                    obj * cls
                }
                else -> {
                    var s = channels[4][j]
                    if (applySigmoidToScore) s = sigmoid(s)
                    s
                }
            }

            if (score < scoreThreshold) continue
            if (coordsAreNormalized) {
                cx *= inputWidth; cy *= inputHeight
                w  *= inputWidth; h  *= inputHeight
            }

            val l = cx - w/2f; val t = cy - h/2f; val r = cx + w/2f; val b = cy + h/2f
            val boxLb = RectF(l, t, r, b)
            val mapped = if (useLetterbox) fromLetterbox(boxLb, pp.scale, pp.dx, pp.dy) else boxLb
            mapped.intersect(0f, 0f, src.width.toFloat(), src.height.toFloat())
            if (mapped.width() <= 0f || mapped.height() <= 0f) continue

            val labelText = labels.getOrNull(0) ?: "object"
            out += YoloDetection(mapped, YoloLabel(labelText, score, 0))
            if (score > maxScoreSeen) maxScoreSeen = score
        }
        Log.i(TAG, "decode5xN: N=$N kept=${out.size} maxScore=${"%.3f".format(maxScoreSeen)} A=$A")
        return out
    }

    // For row-major outputs: [N, A] where A = 5(+C) and order is [cx,cy,w,h,obj,(cls...)]
    private fun decodeRowMajorNxA(
        rows: Array<FloatArray>,
        src: Bitmap,
        pp: PP,
        N: Int,
        A: Int
    ): List<YoloDetection> {
        val out = mutableListOf<YoloDetection>()
        val numClasses = (A - 5).coerceAtLeast(0)
        var maxConf = 0f

        for (i in 0 until N) {
            val row = rows[i]
            var cx = row[0]; var cy = row[1]; var w = row[2]; var h = row[3]
            var obj = row[4]
            obj = sigmoid(obj) // row-major YOLO heads almost always need sigmoid

            var bestIdx = 0
            var bestScore = if (numClasses == 0) 1f else 0f
            if (numClasses > 0) {
                for (c in 0 until numClasses) {
                    val raw = row[5 + c]
                    val p = sigmoid(raw)
                    if (p > bestScore) { bestScore = p; bestIdx = c }
                }
            }
            val conf = obj * bestScore
            if (conf < scoreThreshold) continue
            maxConf = max(maxConf, conf)

            if (coordsAreNormalized) {
                cx *= inputWidth; cy *= inputHeight
                w  *= inputWidth; h  *= inputHeight
            }
            val l = cx - w/2f; val t = cy - h/2f; val r = cx + w/2f; val b = cy + h/2f
            val boxLb = RectF(l, t, r, b)
            val mapped = if (useLetterbox) fromLetterbox(boxLb, pp.scale, pp.dx, pp.dy) else boxLb
            mapped.intersect(0f, 0f, src.width.toFloat(), src.height.toFloat())
            if (mapped.width() <= 0f || mapped.height() <= 0f) continue

            val labelText = labels.getOrNull(bestIdx) ?: "class_$bestIdx"
            out += YoloDetection(mapped, YoloLabel(labelText, conf, bestIdx))
        }
        Log.i(TAG, "decodeNxA: N=$N kept=${out.size} maxConf=${"%.3f".format(maxConf)} A=$A classes=$numClasses")
        return out
    }

    // ------------------ Helpers ------------------

    private data class PP(val letterboxed: Bitmap, val scale: Float, val dx: Float, val dy: Float)

    private fun letterbox(src: Bitmap, dstW: Int, dstH: Int, padColor: Int): PP {
        val scale = min(dstW.toFloat() / src.width, dstH.toFloat() / src.height)
        val newW = (src.width * scale).toInt().coerceAtLeast(1)
        val newH = (src.height * scale).toInt().coerceAtLeast(1)
        val dx = (dstW - newW) / 2f
        val dy = (dstH - newH) / 2f

        val resized = src.scale(newW, newH)
        val out = createBitmap(dstW, dstH)
        val c = Canvas(out)
        c.drawColor(padColor)
        c.drawBitmap(resized, dx, dy, null)
        return PP(out, scale, dx, dy)
    }

    private fun resize(src: Bitmap, dstW: Int, dstH: Int): Bitmap = src.scale(dstW, dstH)

    private fun fromLetterbox(boxLb: RectF, scale: Float, dx: Float, dy: Float): RectF {
        return RectF(
            (boxLb.left   - dx) / scale,
            (boxLb.top    - dy) / scale,
            (boxLb.right  - dx) / scale,
            (boxLb.bottom - dy) / scale,
        )
    }

    private fun bitmapToFloatBuffer(bm: Bitmap, normalizeTo01: Boolean, bgr: Boolean): ByteBuffer {
        val input = ByteBuffer.allocateDirect(1 * bm.width * bm.height * 3 * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(bm.width * bm.height)
        bm.getPixels(pixels, 0, bm.width, 0, 0, bm.width, bm.height)
        val div = if (normalizeTo01) 255f else 1f
        for (y in 0 until bm.height) {
            for (x in 0 until bm.width) {
                val p = pixels[y * bm.width + x]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                if (!bgr) {
                    input.putFloat(r / div); input.putFloat(g / div); input.putFloat(b / div)
                } else {
                    input.putFloat(b / div); input.putFloat(g / div); input.putFloat(r / div)
                }
            }
        }
        input.rewind()
        return input
    }

    private fun sigmoid(x: Float): Float = (1f / (1f + exp(-x)))

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val iw = max(0f, interRight - interLeft)
        val ih = max(0f, interBottom - interTop)
        val inter = iw * ih
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun nms(dets: List<YoloDetection>, iouThresh: Float): List<YoloDetection> {
        val sorted = dets.sortedByDescending { it.label.confidence }.toMutableList()
        val keep = mutableListOf<YoloDetection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            keep += best
            val it = sorted.iterator()
            while (it.hasNext()) {
                val o = it.next()
                // per-class suppression (works even with 1-class)
                if (best.label.index == o.label.index && iou(best.box, o.box) > iouThresh) it.remove()
            }
        }
        return keep
    }

    private fun loadModelFile(ctx: Context, path: String): MappedByteBuffer {
        val afd = ctx.assets.openFd(path)
        FileInputStream(afd.fileDescriptor).use { fis ->
            val channel = fis.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    private fun loadLabels(ctx: Context, path: String): List<String> = try {
        ctx.assets.open(path).bufferedReader().readLines().filter { it.isNotBlank() }
    } catch (_: Exception) {
        emptyList() // single-class OK
    }

    // ------------- Rich debug log -------------

    private fun debugReport(image: Bitmap, detections: List<YoloDetection>) {
        val W = image.width;
        val H = image.height
        val count = detections.size
        Log.i(TAG, "────────────────────────────────────────────")
        Log.i(TAG, "DETECTION REPORT img=${W}x${H}  count=$count")
        if (count == 0) {
            Log.i(TAG, "No detections.")
            Log.i(TAG, "────────────────────────────────────────────")
            return
        }
        val confs = detections.map { it.label.confidence }
        val minC = confs.minOrNull() ?: 0f
        val maxC = confs.maxOrNull() ?: 0f
        val avgC = if (confs.isNotEmpty()) confs.average().toFloat() else 0f
        Log.i(TAG, "Confidence  min=${"%.2f".format(minC)}  max=${"%.2f".format(maxC)}  avg=${"%.2f".format(avgC)}")

        detections.forEachIndexed { i, det ->
            val b = det.box
            val L = b.left.roundToInt(); val T = b.top.roundToInt()
            val R = b.right.roundToInt(); val B = b.bottom.roundToInt()
            val w = (b.right - b.left).coerceAtLeast(0f)
            val h = (b.bottom - b.top).coerceAtLeast(0f)
            val cls = det.label.text
            val conf = det.label.confidence
            Log.i(TAG, "— #$i  $cls  conf=${"%.2f".format(conf)}")
            Log.i(TAG, "   px  L=$L  T=$T  R=$R  B=$B  |  W=${w.roundToInt()}  H=${h.roundToInt()}")
        }
        Log.i(TAG, "────────────────────────────────────────────")
    }
}

