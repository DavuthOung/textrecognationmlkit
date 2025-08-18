package com.example.text_recognation_ml_kit

import android.annotation.SuppressLint
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
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

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

private fun normalizeOcr(raw: String): String {
    // Keep newlines; only squeeze internal spaces; fix common OCR typos.
    val squeezed = raw
        .lines()
        .joinToString("\n") { it.trim().replace(Regex("\\s+"), " ") }
    return squeezed
        .replace(Regex("(?i)lssue"), "Issue")
        .replace(Regex("(?i)Expir[yv]"), "Expiry")
        .replace(Regex("(?i)Cateaories|Cateqories|Calegories"), "Categories")
        .replace(Regex("(?i)CardGode|CardCodee"), "CardCode")
        .replace(Regex("(?i)Natlonality|Natlonalily|Nati0nality"), "Nationality")
        .replace(Regex("(?i)ts\\.mpwt\\.govJth"), "ts.mpwt.gov.kh")
}


@SuppressLint("DefaultLocale")
private fun toIsoDateOrNull(rawDate: String?): String? {
    if (rawDate.isNullOrBlank()) return null
    val parts = rawDate.trim().split(Regex("\\D+"))
    if (parts.size != 3) return null
    val d = parts[0].toIntOrNull() ?: return null
    val m = parts[1].toIntOrNull() ?: return null
    var y = parts[2].toIntOrNull() ?: return null
    if (y < 100) y = if (y >= 50) 1900 + y else 2000 + y
    if (m !in 1..12 || d !in 1..31 || y !in 1900..2100) return null
    return String.format("%04d-%02d-%02d", y, m, d)
}

private fun extractDatesSortedIso(allText: String): List<String> {
    // Why: we only want dates, normalized and chronologically sorted.
    val dateRegex = Regex("([0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4})")

    return dateRegex.findAll(allText)
        .map { toIsoDateOrNull(it.groupValues[1]) }
        .filterNotNull()
        .sorted() as List<String>
}

private fun pickIssueAndExpiry(allText: String, dobIso: String?): Pair<String?, String?> {
    val dateRegex = Regex("(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4})")

    // Explicit both-on-one-line pattern first
    val both = Regex(
        "(?is)\\bIssue\\s*Date\\b.*?(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4}).*?\\bExpiry\\s*Date\\b.*?(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4})"
    ).find(allText)
    if (both != null) {
        val issueIso = toIsoDateOrNull(both.groupValues[1])
        val expiryIso = toIsoDateOrNull(both.groupValues[2])
        return issueIso to expiryIso
    }

    // Separate labels
    val issueIso = Regex("(?is)\\bIssue\\s*Date\\b.*?(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4})")
        .find(allText)?.groupValues?.getOrNull(1)?.let { toIsoDateOrNull(it) }
    val expiryIso = Regex("(?is)\\bExpiry\\s*Date\\b.*?(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4})")
        .find(allText)?.groupValues?.getOrNull(1)?.let { toIsoDateOrNull(it) }
    if (issueIso != null || expiryIso != null) return issueIso to expiryIso

    // Fallback: take all dates except the DoB; choose min as issue, max as expiry.
    val allIso = dateRegex.findAll(allText)
        .map { toIsoDateOrNull(it.groupValues[1]) }
        .filterNotNull()
        .filter { it != dobIso }
        .toList()
        .distinct()
        .sorted()
    if (allIso.size >= 2) return allIso.first() to allIso.last()
    return null to null
}

suspend fun extractCambodianLicenseData(
    context: Context,
    drawableId: Int,
): CambodianDrivingLicense {
    val bitmap: Bitmap = BitmapFactory.decodeResource(context.resources, drawableId)
    val image = InputImage.fromBitmap(bitmap, 0)
    val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    val minConfidence = 0.5f

    return try {
        val visionText: Text = recognizer.process(image).await()

        // Build a filtered text snapshot (guard against older APIs lacking confidence fields).
        val filtered = StringBuilder()
        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                val lineOk = try { line?.confidence ?: 1f } catch (_: Throwable) { 1f } >= minConfidence
                if (!lineOk) continue
                val lineText = buildString {
                    for (el in line.elements) {
                        val elOk = try { el.confidence >= minConfidence } catch (_: Throwable) { true }
                        if (elOk) append(el.text).append(' ')
                    }
                }.trim()
                if (lineText.isNotBlank()) filtered.append(lineText).append('\n')
            }
        }

        val allFilteredText = filtered.toString().trim().ifEmpty { visionText.text }
        val normalized = normalizeOcr(allFilteredText)

        // --- Regex parsing ---
        val idNumber = Regex("(?i)\\bID\\b\\s*[:：-]?\\s*([A-Z0-9]{6,})").find(normalized)?.groupValues?.getOrNull(1)

        val cardCode = Regex("(?i)\\bCard\\s*Code\\b\\s*[:：-]?\\s*([A-Z]\\.[A-Z]{1,3}\\.[0-9]{5,})")
            .find(normalized)?.groupValues?.getOrNull(1)
            ?: Regex("\\b([A-Z]\\.[A-Z]{1,3}\\.[0-9]{5,})\\b").find(normalized)?.groupValues?.getOrNull(1)

        val category = Regex("(?is)(?:\\bCategories?\\b).*?(?:\\n|\\s)([A-EDM](?:\\s*[,&/]\\s*[A-EDM])*)")
            .find(normalized)?.groupValues?.getOrNull(1)?.trim()?.replace(" ", "")

        val fullName = Regex("(?is)\\bSurname\\s*&\\s*Name\\b\\s*(?:\\n|\\s)+([A-Z][A-Z\\s]+)")
            .find(normalized)?.groupValues?.getOrNull(1)?.trim()
        val (surnameEnglish, givenNameEnglish) = if (!fullName.isNullOrBlank()) {
            val parts = fullName.split(Regex("\\s+")).filter { it.isNotBlank() }
            if (parts.size >= 2) parts.first() to parts.drop(1).joinToString(" ") else null to fullName
        } else null to null

        val sexRaw = Regex("(?is)\\bSex\\b\\s*(?:\\n|\\s)*([MF]|Male|Female)\\b")
            .find(normalized)?.groupValues?.getOrNull(1)
        val sex = when (sexRaw?.uppercase()) { "MALE", "M" -> "M"; "FEMALE", "F" -> "F"; else -> null }

        val nationality = Regex("(?is)\\bNationality\\b\\s*(?:\\n|\\s)*([A-Za-z][A-Za-z ]+)")
            .find(normalized)?.groupValues?.getOrNull(1)?.trim()?.split(" ")?.firstOrNull()
            ?.replaceFirstChar { it.uppercase() }

        val dobRaw = Regex("(?is)\\bDate\\s*Of\\s*Birth\\b.*?(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4})")
            .find(normalized)?.groupValues?.getOrNull(1)
        val dobIso = toIsoDateOrNull(dobRaw)

        val (issueIso, expiryIso) = pickIssueAndExpiry(normalized, dobIso)

        Log.d("LicenseParse", "RAW\n$allFilteredText\n\nNORMALIZED\n$normalized")

        CambodianDrivingLicense(
            surnameEnglish = surnameEnglish,
            givenNameEnglish = givenNameEnglish,
            idNumber = idNumber,
            dateOfBirth = dobIso,
            sex = sex,
            nationality = nationality,
            address = null,
            placeOfBirth = null,
            dateOfIssue = issueIso,
            dateOfExpiry = expiryIso,
            category = category,
            cardCode = cardCode,
            rawText = allFilteredText
        )
    } catch (e: Exception) {
        Log.e("LicenseParse", "Error parsing license data", e)
        CambodianDrivingLicense(rawText = "Error: ${e.message}")
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


@Preview(showBackground = true)
@Composable
fun TextRecognitionScreenPreview() {
    TextrecognationmlkitTheme {
        TextRecognitionScreen()
    }
}
