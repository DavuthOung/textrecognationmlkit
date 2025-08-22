package com.example.text_recognation_ml_kit

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Enhanced OCR-friendly image preprocessing using OpenCV.
 *
 * This pipeline applies several common techniques to improve Tesseract OCR accuracy.
 *
 * @param inputBgr The input image in BGR format (OpenCV Mat). This is typically a cropped region
 *                 containing the text of interest.
 * @param targetDpi Approximate target DPI to aim for via upscaling. If 0, uses upscaleFactor.
 * @param upscaleFactor Factor for direct upscaling if targetDpi is 0.
 *                      Set to 1.0 or less to disable if targetDpi is also 0.
 * @param estimatedSourceDpi Estimated DPI of the input image. Used with targetDpi for scaling.
 * @param denoiseStrength Strength for denoising (h parameter for fastNlMeansDenoising).
 *                        Set to 0f to disable denoising.
 * @param binarizationMethod Method for binarization: "OTSU", "ADAPTIVE_GAUSSIAN", "NONE".
 * @param adaptiveBlockSize Block size for adaptive thresholding.
 * @param adaptiveC Constant subtracted from the mean for adaptive thresholding.
 * @param attemptDeskew If true, tries to detect and correct skew.
 * @param deskewMaxAngle Max angle (degrees) to attempt deskewing.
 * @return A new Mat object with the preprocessed image, typically grayscale or binary, in BGR format
 *         (as Tesseract input often expects 3 channels, even if effectively grayscale).
 */
suspend fun preprocessForOcrAdvanced(
	inputBgr: Mat,
	targetDpi: Int = 300,
	upscaleFactor: Double = 1.5, // Used if targetDpi is 0 or sourceDpi is unknown
	estimatedSourceDpi: Int = 72, // Common screen DPI, adjust if known
	denoiseStrength: Float = 10f, // 0 to disable, 5-15 is typical
	binarizationMethod: String = "OTSU", // "OTSU", "ADAPTIVE_GAUSSIAN", "NONE"
	adaptiveBlockSize: Int = 11, // Must be odd
	adaptiveC: Double = 2.0,
	attemptDeskew: Boolean = true,
	deskewMaxAngle: Double = 15.0
): Mat = withContext(Dispatchers.Default){
	if (inputBgr.empty()) {
		Log.e("PreprocessOCR", "Input image is empty.")
		return@withContext Mat() // Return empty Mat from the coroutine context
	}

	// 1. Convert to Grayscale
	val gray = Mat()
	Imgproc.cvtColor(inputBgr, gray, Imgproc.COLOR_BGR2GRAY)
	var currentMat = gray // Keep track of the latest processed Mat

	// 2. Upscaling (if necessary)
	val scale: Double =
		if (targetDpi > 0 && estimatedSourceDpi > 0 && targetDpi > estimatedSourceDpi) {
			targetDpi.toDouble() / estimatedSourceDpi.toDouble()
		} else if (upscaleFactor > 1.0) {
			upscaleFactor
		} else {
			1.0
		}

	if (scale > 1.0) {
		val upscaled = Mat()
		val newWidth = max(1.0, currentMat.cols() * scale).toInt()
		val newHeight = max(1.0, currentMat.rows() * scale).toInt()
		Imgproc.resize(
			currentMat,
			upscaled,
			Size(newWidth.toDouble(), newHeight.toDouble()),
			0.0,
			0.0,
			Imgproc.INTER_CUBIC
		)
		currentMat.release()
		currentMat = upscaled
		Log.d("PreprocessOCR", "Upscaled by factor: $scale to ${currentMat.size()}")
	}

	// 3. Deskewing (Optional but can be very effective)
	if (attemptDeskew) {
		val deskewed = Mat()
		val angle = computeSkewAngle(currentMat, deskewMaxAngle)
		if (abs(angle) > 0.1) { // Only rotate if skew is significant
			Log.d("PreprocessOCR", "Deskewing. Detected angle: $angle degrees")
			val center = Point(currentMat.cols() / 2.0, currentMat.rows() / 2.0)
			val rotationMatrix = Imgproc.getRotationMatrix2D(center, angle, 1.0)

			// Calculate bounding box for the rotated image to avoid cropping
			val absCos = abs(cos(Math.toRadians(angle)))
			val absSin = abs(sin(Math.toRadians(angle)))
			val newWidth = (currentMat.height() * absSin + currentMat.width() * absCos).toInt()
			val newHeight = (currentMat.height() * absCos + currentMat.width() * absSin).toInt()

			// Adjust translation in rotation matrix
			rotationMatrix.put(0, 2, rotationMatrix.get(0, 2)[0] + (newWidth / 2.0 - center.x))
			rotationMatrix.put(1, 2, rotationMatrix.get(1, 2)[0] + (newHeight / 2.0 - center.y))

			Imgproc.warpAffine(
				currentMat,
				deskewed,
				rotationMatrix,
				Size(newWidth.toDouble(), newHeight.toDouble()),
				Imgproc.INTER_CUBIC,
				Core.BORDER_REPLICATE
			)
			currentMat.release()
			currentMat = deskewed
			rotationMatrix.release()
		} else {
			deskewed.release() // Not used
			Log.d("PreprocessOCR", "No significant skew detected or deskewing skipped.")
		}
	}


	// 4. Denoising (Optional)
	if (denoiseStrength > 0f) {
		val denoised = Mat()
		Photo.fastNlMeansDenoising(currentMat, denoised, denoiseStrength, 7, 21)
		currentMat.release()
		currentMat = denoised
		Log.d("PreprocessOCR", "Denoising applied with strength: $denoiseStrength")
	}

	// 5. Binarization / Thresholding
	val binarized = Mat()
	when (binarizationMethod.uppercase()) {
		"OTSU" -> {
			Imgproc.threshold(
				currentMat,
				binarized,
				0.0,
				255.0,
				Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU
			)
			Log.d("PreprocessOCR", "Binarization: OTSU applied.")
		}

		"ADAPTIVE_GAUSSIAN" -> {
			Imgproc.adaptiveThreshold(
				currentMat, binarized, 255.0,
				Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY,
				if (adaptiveBlockSize % 2 == 0) adaptiveBlockSize + 1 else adaptiveBlockSize, // Ensure odd block size
				adaptiveC
			)
			Log.d("PreprocessOCR", "Binarization: ADAPTIVE_GAUSSIAN applied.")
		}

		"NONE" -> {
			currentMat.copyTo(binarized) // No binarization, keep as grayscale
			Log.d("PreprocessOCR", "Binarization: NONE (kept as grayscale).")
		}

		else -> {
			Log.w(
				"PreprocessOCR",
				"Unknown binarization method: $binarizationMethod. Using OTSU as default."
			)
			Imgproc.threshold(
				currentMat,
				binarized,
				0.0,
				255.0,
				Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU
			)
		}
	}
	currentMat.release()
	currentMat = binarized


	// 6. Optional: Morphological Operations (e.g., to thicken thin characters or remove small noise)
	// This is highly dependent on the text characteristics.
	// Example: Opening to remove small noise, or Dilation to thicken characters
	// val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.5, 1.5))
	// Imgproc.morphologyEx(currentMat, currentMat, Imgproc.MORPH_OPEN, kernel)
	// kernel.release()

	// 7. Convert final processed image back to BGR if it's not already
	// Tesseract can handle grayscale, but sometimes providing a 3-channel image where R=G=B
	// can be more stable with certain Tesseract versions or configurations.
	// If currentMat is already 3-channel (e.g., if no processing was done after initial BGR), this is not needed.
	// If it's single-channel (grayscale/binary), convert it.
	val finalOutputBgr = Mat()
	if (currentMat.channels() == 1) {
		Imgproc.cvtColor(currentMat, finalOutputBgr, Imgproc.COLOR_GRAY2BGR)
		currentMat.release()
	} else {
		currentMat.copyTo(finalOutputBgr) // Already BGR or processed into a 3-channel format
		currentMat.release() // or just currentMat.release() if you assign finalOutputBgr = currentMat
	}

	Log.d(
		"PreprocessOCR",
		"Preprocessing complete. Final image size: ${finalOutputBgr.size()}, channels: ${finalOutputBgr.channels()}"
	)
	finalOutputBgr
}

/**
 * Computes the skew angle of an image containing text.
 * This is a simplified version using Hough Line Transform.
 * More robust methods might involve contour analysis or projection profiles.
 *
 * @param grayImage Input grayscale image.
 * @param maxAngleToDetect Max angle (in degrees, positive or negative) to consider for deskewing.
 * @return The dominant skew angle in degrees. Negative for counter-clockwise, positive for clockwise.
 */
private fun computeSkewAngle(grayImage: Mat, maxAngleToDetect: Double = 15.0): Double {
	val size = grayImage.size()
	val edges = Mat()

	// Invert the image if text is dark on light background, as HoughLinesP works better with white lines on black.
	// If your binarization already ensures white text on black background, this might not be needed.
	val inverted = Mat()
	Core.bitwise_not(
		grayImage,
		inverted
	) // Assuming grayImage is binary (white text on black or vice-versa)
	// If not binary, Otsu might be needed here or ensure input is binary.

	Imgproc.Canny(inverted, edges, 50.0, 200.0, 3, false) // Adjust Canny thresholds

	val lines = Mat() // Output of HoughLinesP
	val thresholdHough = max(size.width, size.height).toInt() / 10 // Heuristic threshold
	val minLineLength: Double = (max(size.width, size.height).toInt() / 20).toDouble()
	val maxLineGap: Double = (max(size.width, size.height).toInt() / 100).toDouble()

	Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180, thresholdHough, minLineLength, maxLineGap)

	var totalAngle = 0.0
	var count = 0

	for (i in 0 until lines.rows()) {
		val vec = lines.get(i, 0)
		val x1 = vec[0]
		val y1 = vec[1]
		val x2 = vec[2]
		val y2 = vec[3]

		var angle = Math.toDegrees(atan2(y2 - y1, x2 - x1))

		// Filter angles to be within a reasonable range for text skew
		if (abs(angle) <= maxAngleToDetect || abs(abs(angle) - 90) <= maxAngleToDetect || abs(
				abs(
					angle
				) - 180
			) <= maxAngleToDetect
		) {
			// Normalize angles to be close to 0
			if (abs(angle) > 45 && abs(angle) < 135) { // Likely near +/- 90 degrees (vertical text or lines)
				angle = if (angle > 0) angle - 90 else angle + 90
			} else if (abs(angle) >= 135) { // Likely near +/- 180
				angle = if (angle > 0) angle - 180 else angle + 180
			}

			if (abs(angle) <= maxAngleToDetect) { // Final check after normalization
				totalAngle += angle
				count++
			}
		}
	}

	lines.release()
	edges.release()
	inverted.release()

	return if (count > 0) totalAngle / count else 0.0
}