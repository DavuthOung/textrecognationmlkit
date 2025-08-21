package com.example.text_recognation_ml_kit

import android.graphics.Bitmap
import android.util.Log
import com.googlecode.tesseract.android.TessBaseAPI
import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream

object TessDataManager {

	private const val TAG = "TessDataManager"
	// This is the directory on the DEVICE (inside your app's filesDir)
	// that will contain the "tessdata" subdirectory.
	// This is the path Tesseract's init() method needs.
	const val TESSERACT_MODELS_PARENT_DIR_ON_DEVICE = "myTessModels"
	const val TESSDATA_SUBDIR_NAME = "tessdata" // Standard Tesseract subdirectory name

	// Path within ASSETS where your .traineddata files are, RELATIVE TO "tessdata"
	// So if it's assets/models/tessdata/ocrb.traineddata, this path is "models/tessdata"
	const val ASSET_TESSDATA_BASE_PATH = "models/tessdata"


	/**
	 * Returns the path to the directory ON THE DEVICE that will contain the "tessdata" subdirectory.
	 * This is the path Tesseract's init() method needs.
	 * e.g., /data/user/0/your.package.name/files/myTessModels
	 */
	fun getTesseractParentDataPathOnDevice(context: Context): String {
		return File(context.filesDir, TESSERACT_MODELS_PARENT_DIR_ON_DEVICE).absolutePath
	}

	/**
	 * Returns the full path to where the "tessdata" directory should be on the device.
	 * e.g., /data/user/0/your.package.name/files/myTessModels/tessdata
	 */
	private fun getDeviceTessdataDirPath(context: Context): String {
		return File(getTesseractParentDataPathOnDevice(context), TESSDATA_SUBDIR_NAME).absolutePath
	}


	/**
	 * Checks if the language data file exists in the app's private storage.
	 * @param language e.g., "ocrb", "eng"
	 */
	fun isLanguageDataAvailable(context: Context, language: String): Boolean {
		val deviceTessdataDir = File(getDeviceTessdataDirPath(context))
		val languageFile = File(deviceTessdataDir, "$language.traineddata")
		return languageFile.exists() && languageFile.length() > 0
	}

	/**
	 * Copies the specified language data file from "assets/models/tessdata/"
	 * to app's private storage at "[app_files_dir]/myTessModels/tessdata/".
	 * @param language e.g., "ocrb", "eng"
	 * @param overwrite if true, will overwrite existing file
	 */
	fun copyLanguageDataToStorage(context: Context, language: String, overwrite: Boolean = false): Boolean {
		val tesseractParentDirOnDevice = getTesseractParentDataPathOnDevice(context)
		val deviceModelsDir = File(tesseractParentDirOnDevice) // e.g., .../files/myTessModels
		val deviceTessdataDir = File(deviceModelsDir, TESSDATA_SUBDIR_NAME) // e.g., .../files/myTessModels/tessdata

		// Create the parent "myTessModels" directory on device if it doesn't exist
		if (!deviceModelsDir.exists()) {
			if (!deviceModelsDir.mkdirs()) {
				Log.e(TAG, "Failed to create directory: ${deviceModelsDir.absolutePath}")
				return false
			}
		}

		// Create the "tessdata" subdirectory inside "myTessModels" on device if it doesn't exist
		if (!deviceTessdataDir.exists()) {
			if (!deviceTessdataDir.mkdirs()) {
				Log.e(TAG, "Failed to create directory: ${deviceTessdataDir.absolutePath}")
				return false
			}
		}

		val trainedDataFileOnDevice = File(deviceTessdataDir, "$language.traineddata")

		if (trainedDataFileOnDevice.exists() && !overwrite && trainedDataFileOnDevice.length() > 0) {
			Log.i(TAG, "$language.traineddata already exists at ${trainedDataFileOnDevice.absolutePath}. Skipping copy.")
			return true
		}

		// Correct asset path: assets/models/tessdata/ocrb.traineddata
		val assetPath = "$ASSET_TESSDATA_BASE_PATH/$language.traineddata"
		try {
			Log.d(TAG, "Attempting to copy from assets: $assetPath to ${trainedDataFileOnDevice.absolutePath}")
			context.assets.open(assetPath).use { inputStream ->
				FileOutputStream(trainedDataFileOnDevice).use { outputStream ->
					copyFile(inputStream, outputStream)
					Log.i(TAG, "Copied $language.traineddata to ${trainedDataFileOnDevice.absolutePath}")
					return true
				}
			}
		} catch (e: IOException) {
			Log.e(TAG, "Error copying $language.traineddata from assets '$assetPath': ${e.message}", e)
			if (trainedDataFileOnDevice.exists()) {
				trainedDataFileOnDevice.delete()
			}
			return false
		}
	}

	@Throws(IOException::class)
	private fun copyFile(inputStream: InputStream, outputStream: OutputStream) {
		val buffer = ByteArray(1024 * 4)
		var read: Int
		while (inputStream.read(buffer).also { read = it } != -1) {
			outputStream.write(buffer, 0, read)
		}
		outputStream.flush()
	}

	fun ensureTesseractData(context: Context, languages: List<String>) {
		Log.d(TAG, "Ensuring Tesseract data for languages: $languages")
		languages.forEach { lang ->
			if (!isLanguageDataAvailable(context, lang)) {
				Log.i(TAG, "Language data for '$lang' not found. Attempting to copy...")
				if (copyLanguageDataToStorage(context, lang)) {
					Log.i(TAG, "Successfully copied '$lang' data.")
				} else {
					Log.e(TAG, "Failed to copy '$lang' data.")
				}
			} else {
				Log.i(TAG, "Language data for '$lang' is available.")
			}
		}
	}
}


class Tesseract {

	var tessBaseAPI: TessBaseAPI? = null
	private var currentLanguage: String? = null // Keep track of current language
	val context: Context
	private val TAG = "TesseractHelper"


	constructor(context: Context, defaultLanguage: String = "ocrb") { // Default to "eng" or your common lang
		this.context = context.applicationContext // Use application context
		// Ensure data is copied before attempting to initialize
		// This is crucial. It should ideally be done once at app startup
		// or guaranteed to be done before Tesseract object is created.
		TessDataManager.ensureTesseractData(this.context, listOf(defaultLanguage))
		initializeTesseract(defaultLanguage)
	}


	fun initializeTesseract(language: String): Boolean {
		// ... (existing checks and cleanup) ...

		tessBaseAPI = TessBaseAPI()
		currentLanguage = null

		// This is the path ON THE DEVICE to the directory that CONTAINS the "tessdata" folder
		// e.g., /data/user/0/your.package.name/files/myTessModels/
		val parentDataPathOnDevice = TessDataManager.getTesseractParentDataPathOnDevice(context)

		// Verify that the "tessdata" subdirectory actually exists inside parentDataPathOnDevice
		val deviceTessdataDir = File(parentDataPathOnDevice, TessDataManager.TESSDATA_SUBDIR_NAME)
		if (!deviceTessdataDir.exists() || !deviceTessdataDir.isDirectory) {
			Log.e(TAG, "CRITICAL: Tessdata directory not found at: ${deviceTessdataDir.absolutePath}. " +
					"Ensure language files were copied correctly from 'assets/${TessDataManager.ASSET_TESSDATA_BASE_PATH}'.")
			// Attempt to copy again as a last resort
			TessDataManager.ensureTesseractData(context, listOf(language))
			if (!deviceTessdataDir.exists() || !deviceTessdataDir.isDirectory) {
				tessBaseAPI?.recycle()
				tessBaseAPI = null
				Log.e(TAG, "Still cannot find tessdata directory after retry. Aborting initialization.")
				return false
			}
		}

		Log.d(TAG, "Attempting to initialize Tesseract with language: '$language', parentDataPathOnDevice: '$parentDataPathOnDevice'")

		if (tessBaseAPI!!.init(parentDataPathOnDevice, language)) {
			Log.i(TAG, "Tesseract initialized successfully for '$language'.")
			currentLanguage = language
			return true
		} else {
			Log.e(TAG, "Failed to initialize Tesseract for '$language' with parentDataPathOnDevice '$parentDataPathOnDevice'. " +
					"Check Tesseract internal logs (Logcat tag 'Tesseract'). Ensure '$language.traineddata' exists in '${deviceTessdataDir.absolutePath}'.")
			tessBaseAPI!!.recycle()
			tessBaseAPI = null
			return false
		}
	}


	fun extractText(bitmap: Bitmap, language: String = "ocrb"): String? { // Allow specifying language
		if (tessBaseAPI == null || currentLanguage != language) {
			Log.w(TAG, "Tesseract not initialized for '$language' or language changed. Attempting to re-initialize.")
			if (!initializeTesseract(language)) {
				Log.e(TAG, "Failed to initialize for '$language'. Cannot extract text.")
				return null // Or throw an exception
			}
		}

		// Check again after re-initialization attempt
		if (tessBaseAPI == null) {
			Log.e(TAG, "Tesseract is still null after re-initialization attempt for '$language'.")
			return null
		}


		tessBaseAPI!!.setImage(bitmap)
		val text = tessBaseAPI!!.utF8Text
		Log.d(TAG, "Extracted text for language '$language': ${text?.take(100)}...") // Log a snippet
		return text
	}

	fun destroy() {
		tessBaseAPI?.recycle()
		tessBaseAPI = null
		currentLanguage = null
		Log.i(TAG, "Tesseract instance destroyed.")
	}
}
