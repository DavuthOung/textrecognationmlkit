package com.example.text_recognation_ml_kit

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.text_recognation_ml_kit.ui.theme.TextrecognationmlkitTheme
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

data class CambodianDrivingLicense(
    val surnameEnglish: String? = null,
    val givenNameEnglish: String? = null,
    val idNumber: String? = null,
    val dateOfBirth: String? = null,
    val sex: String? = null,
    val nationality: String? = null,
    val address: String? = null,
    val placeOfBirth: String? = null,
    val dateOfIssue: String? = null,
    val dateOfExpiry: String? = null,
    val category: String? = null,
    val cardCode: String? = null,
    val rawText: String = ""
)

suspend fun extractCambodianLicenseData(context: Context, drawableId: Int): CambodianDrivingLicense {
    val bitmap: Bitmap = BitmapFactory.decodeResource(context.resources, drawableId)
    val image = InputImage.fromBitmap(bitmap, 0)
    val minConfidence = 0.5f

    try {
        val visionText = recognizer.process(image).await()
        val filteredTextBuilder = StringBuilder()

        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                if (line.confidence >= minConfidence) {
                    val lineText = line.elements.filter { it.confidence >= minConfidence }
                        .joinToString(" ") { it.text }
                    if (lineText.isNotBlank()) filteredTextBuilder.append(lineText).append("\n")
                }
            }
        }

        val allFilteredText = filteredTextBuilder.toString().trim()
        val textLines = allFilteredText.split('\n').map { it.trim() }

        // Regex
        val datePattern = Regex("(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4})")
        val idPattern = Regex("ID\\s*[:：]?\\s*([A-Z0-9]+)")
        val cardCodePattern = Regex("([ABC]\\.[0-9]+)")
        val categoryPattern = Regex("([ABCDE])")
        var surnameEnglish: String? = null
        var givenNameEnglish: String? = null
        var idNumber: String? = null
        var dob: String? = null
        var sex: String? = null
        var nationality: String? = null
        var address: String? = null
        var placeOfBirth: String? = null
        var issueDate: String? = null
        var expiryDate: String? = null
        var category: String? = null
        var cardCode: String? = null

        for (i in textLines.indices) {
            val line = textLines[i]
            Log.d("MLKitTextRecognition", "Line $i: $line")
            Log.d("MLKitTextRecognition", "Line $i: $line")
            if (idNumber == null && idPattern.containsMatchIn(line)) idNumber = idPattern.find(line)?.groupValues?.get(1)
            if (line.startsWith("Surname", true)) {
                val name = extractValueAfterKeyword(line, "Surname & Name") ?: textLines.getOrNull(i+1)
                if (!name.isNullOrBlank()) {
                    val parts = name.split(" ", limit = 2)
                    surnameEnglish = parts.getOrNull(0)
                    givenNameEnglish = parts.getOrNull(1)
                }
            }
            if (line.startsWith("Date Of Birth", true)) dob = datePattern.find(line)?.value ?: textLines.getOrNull(i+1)?.let { datePattern.find(it)?.value }
            if (line.startsWith("Sex", true)) sex = extractValueAfterKeyword(line, "Sex") ?: textLines.getOrNull(i+1)
            if (line.startsWith("Nationality", true)) nationality = extractValueAfterKeyword(line, "Nationality") ?: textLines.getOrNull(i+1)
            if (line.startsWith("Address", true)) address = extractValueAfterKeyword(line, "Address") ?: textLines.getOrNull(i+1)
            if (line.startsWith("Place Of Birth", true)) placeOfBirth = extractValueAfterKeyword(line, "Place Of Birth") ?: textLines.getOrNull(i+1)
            if (line.contains("Issue Date", true)) issueDate = datePattern.find(line)?.value ?: textLines.getOrNull(i+1)?.let { datePattern.find(it)?.value }
            if (line.contains("Expiry Date", true)) expiryDate = datePattern.find(line)?.value ?: textLines.getOrNull(i+1)?.let { datePattern.find(it)?.value }
            if (line.startsWith("Categories", true)) {
                category = categoryPattern.find(line)?.value
                if (category.isNullOrBlank()) category = textLines.getOrNull(i+1)?.let { categoryPattern.find(it)?.value }
            }
            if (cardCode == null && cardCodePattern.containsMatchIn(line)) cardCode = cardCodePattern.find(line)?.groupValues?.get(1)
        }

        return CambodianDrivingLicense(
            surnameEnglish = surnameEnglish,
            givenNameEnglish = givenNameEnglish,
            idNumber = idNumber,
            dateOfBirth = dob,
            sex = sex,
            nationality = nationality,
            address = address,
            placeOfBirth = placeOfBirth,
            dateOfIssue = issueDate,
            dateOfExpiry = expiryDate,
            category = category,
            cardCode = cardCode,
            rawText = allFilteredText
        )

    } catch (e: Exception) {
        Log.e("LicenseParse", "Error parsing license data", e)
        return CambodianDrivingLicense(rawText = "Error: ${e.message}")
    }
}


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            TextrecognationmlkitTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    TextRecognitionScreen()
                }
            }
        }


    }
}


@Composable
fun TextRecognitionScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var recognizedText by remember { mutableStateOf("Tap 'Recognize Text' to start.") }
    var isLoading by remember { mutableStateOf(false) }

    val imageToRecognize = R.drawable.test2

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(top = 40.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Top
    ) {
        Button(
            onClick = {
                if (isLoading) return@Button // Prevent multiple clicks while loading

                isLoading = true
                recognizedText = "Processing..." // Update UI
                scope.launch {
                     try {
                         isLoading = true
                         // Clear previous data

                         val result = extractCambodianLicenseData(context, imageToRecognize)
                         // You can format the display text from the licenseData object
                         recognizedText = """
                             Surname (En): ${result.surnameEnglish ?: "N/A"}
                             Given Name (EN): ${result.givenNameEnglish ?: "N/A"}
                             Sex: ${result.sex ?: "N/A"}
                             ID No: ${result.idNumber ?: "N/A"}
                             DOB: ${result.dateOfBirth ?: "N/A"}
                             Address: ${result.address ?: "N/A"}
                             Expiry: ${result.dateOfExpiry ?: "N/A"}
                             Issue: ${result.dateOfIssue ?: "N/A"}
                             Category: ${result.category ?: "N/A"}
                             Card Code: ${result.cardCode ?: "N/A"}
                             Nationality: ${result.nationality ?: "N/A"}
                             ---
                             Raw Filtered Text:
                             ${result.rawText}
                         """.trimIndent()
                    } catch (e: Exception) {
                        recognizedText = "Error parsing license: ${e.localizedMessage}"
                    } finally {
                        isLoading = false
                    }
                }
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isLoading
        ) {
            Text("Recognize Text from Static Image")
        }

        Spacer(modifier = Modifier.height(16.dp))

        if (isLoading) {
            CircularProgressIndicator()
            Spacer(modifier = Modifier.height(16.dp))
        }

        Text(
            text = "Recognized Text:",
            style = MaterialTheme.typography.titleMedium
        )
        Spacer(modifier = Modifier.height(8.dp))
        // Scrollable area for potentially long recognized text
        Column(modifier = Modifier
            .weight(1f) // Takes remaining space
            .fillMaxWidth()
            .verticalScroll(rememberScrollState()) // Make it scrollable
            .padding(25.dp)
        ) {
            Text(
                text = recognizedText,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

// Helper function to extract value after a keyword on the same line
fun extractValueAfterKeyword(line: String, keyword: String, allowPartialKeywordMatch: Boolean = false): String? {
    val keywordIndex = if (allowPartialKeywordMatch) line.indexOf(keyword) else line.lastIndexOf(keyword) // lastIndexOf for full keyword match
    if (keywordIndex != -1) {
        var value = line.substring(keywordIndex + keyword.length).trim()
        // Remove common separators like ":" or "-" if they are right after the keyword
        if (value.startsWith(":") || value.startsWith("-")) {
            value = value.substring(1).trim()
        }
        return value.ifBlank { null }
    }
    return null
}


@Preview(showBackground = true)
@Composable
fun TextRecognitionScreenPreview() {
    TextrecognationmlkitTheme {
        TextRecognitionScreen()
    }
}
