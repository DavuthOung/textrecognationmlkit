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
import com.google.mlkit.vision.text.Text as MlText

private fun snapshotFromTokens(tokens: List<OcrToken>): String =
    tokens.groupBy { it.lineId }
        .toSortedMap()
        .values
        .joinToString("\n") { line -> line.sortedBy { it.cx }.joinToString(" ") { it.text } }
        .trim()


private fun parseFieldsNoRegex(visionText: MlText): CambodianDrivingLicense {
    val tokens = tokensFrom(visionText)
    val rawText = snapshotFromTokens(tokens)

    if (tokens.isEmpty()) return CambodianDrivingLicense(rawText = visionText.text)
    val bands = lineBands(tokens)

    var nameFull: String? = null
    var sex: String? = null
    var nationality: String? = null
    var dobIso: String? = null
    var issueIso: String? = null
    var expiryIso: String? = null
    var idNumber: String? = null
    var category: String? = null
    var cardCode: String? = null
    var address: String? = null
    var pob: String? = null
    var special: String? = null

    // 1) find labels & pick values to the right (or the next line below)
    val labelHits = mutableListOf<Pair<String, OcrToken>>()
    for (t in tokens) {
        val key = bestLabelKeyFor(t.text)
        if (key != null) labelHits += key to t
    }

    for ((key, tok) in labelHits) {
        val raw = collectRightOrBelow(tok, tokens, bands)
        if (raw.isBlank()) continue
        val rawClean = normValue(collapseSpaces(collapseDelimsToSpaces(raw)))

        when (key) {
            "name" -> if (nameFull.isNullOrBlank()) nameFull = rawClean
            "sex" -> {
                val v = rawClean.uppercase()
                sex = when {
                    v.startsWith("M") -> "M"
                    v.startsWith("F") -> "F"
                    else -> sex
                }
            }
            "nationality" -> nationality = rawClean.split(' ').firstOrNull()?.replaceFirstChar { it.uppercase() }
            "dob" -> dobIso = tryParseDateIso(rawClean) ?: dobIso
            "issue" -> issueIso = tryParseDateIso(rawClean) ?: issueIso
            "expiry" -> expiryIso = tryParseDateIso(rawClean) ?: expiryIso
            "id" -> {
                // pick the longest compact token as ID
                val cand = rawClean.split(' ').maxByOrNull { token -> token.count { it.isLetterOrDigit() } }
                if (!cand.isNullOrBlank() && cand.count { it.isLetterOrDigit() } >= 6) idNumber = cand
            }
            "category" -> {
                val toks = rawClean.split(' ').map { it.trim() }.filter { it.isNotBlank() }
                val cats = toks.filter { isLikelyCategoryToken(it) }
                if (cats.isNotEmpty()) category = cats.joinToString(",")
            }
            "cardCode" -> {
                val pieces = rawClean.split(' ').filter { looksLikeCardCodeToken(it) }
                if (pieces.isNotEmpty()) cardCode = pieces.first()
            }
            "address" -> if (address.isNullOrBlank()) address = rawClean
            "pob" -> if (pob.isNullOrBlank()) pob = rawClean
            "special" -> if (special.isNullOrBlank()) special = rawClean
        }
    }

    // 2) Global fallbacks for card code / categories / dates if labels were missed
    if (cardCode.isNullOrBlank()) {
        val candidate = tokens.map { it.text }.firstOrNull { looksLikeCardCodeToken(it) }
        if (candidate != null) cardCode = candidate
    }

    if (category.isNullOrBlank()) {
        val cats = tokens.map { it.text }.filter { isLikelyCategoryToken(it) }
        if (cats.isNotEmpty()) category = cats.distinct().joinToString(",")
    }

    if (issueIso == null || expiryIso == null) {
        val dateCandidates = tokens.map { it.text }
            .map { it.replace(',', ' ').replace('|', ' ') }
            .filter { t -> t.any { it.isDigit() } && (t.contains('-') || t.contains('/') || t.contains('.')) }
            .mapNotNull { tryParseDateIso(it) }
            .distinct()
            .sorted()
        val filtered = if (dobIso != null) dateCandidates.filterNot { it == dobIso } else dateCandidates
        if (filtered.size >= 2) {
            if (issueIso == null) issueIso = filtered.first()
            if (expiryIso == null) expiryIso = filtered.last()
        }
    }

    val (surnameEn, givenEn) = if (!nameFull.isNullOrBlank()) splitName(nameFull!!) else (null to null)

    return CambodianDrivingLicense(
        surnameEnglish = surnameEn,
        givenNameEnglish = givenEn,
        idNumber = idNumber,
        dateOfBirth = dobIso,
        sex = sex,
        nationality = nationality,
        address = address,
        placeOfBirth = pob,
        dateOfIssue = issueIso,
        dateOfExpiry = expiryIso,
        category = category,
        cardCode = cardCode,
        rawText = rawText
    )
}

/**
 * NOTE: ML Kit Text Recognition v2 does NOT support Khmer script.
 * If you need Khmer lines (names/addresses), consider Cloud Vision with languageHints=["km"]
 * or bundling a Khmer Tesseract model. This parser focuses on English fields on the card.
 */

data class CambodianDrivingLicense(
    val surnameEnglish: String? = null,
    val givenNameEnglish: String? = null,
    val idNumber: String? = null,
    val dateOfBirth: String? = null,   // ISO yyyy-MM-dd
    val sex: String? = null,           // "M" | "F"
    val nationality: String? = null,
    val address: String? = null,
    val placeOfBirth: String? = null,
    val dateOfIssue: String? = null,   // ISO yyyy-MM-dd
    val dateOfExpiry: String? = null,  // ISO yyyy-MM-dd
    val category: String? = null,      // e.g. "A1,B"
    val cardCode: String? = null,      // e.g. "P.B.12345"
    val rawText: String = ""
)

suspend fun extractCambodianLicenseData(
    context: Context,
    drawableId: Int,
): CambodianDrivingLicense {
    // (Optional) use your original decode; or the safer one below
    fun decodeBitmapSafely(context: Context, drawableId: Int, maxDim: Int = 2048): Bitmap {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeResource(context.resources, drawableId, opts)
        var inSample = 1
        val w = opts.outWidth; val h = opts.outHeight
        while (w / inSample > maxDim || h / inSample > maxDim) inSample *= 2
        val load = BitmapFactory.Options().apply { inSampleSize = inSample.coerceAtLeast(1) }
        return BitmapFactory.decodeResource(context.resources, drawableId, load)
    }

    val bitmap: Bitmap = decodeBitmapSafely(context, drawableId)
    val image = InputImage.fromBitmap(bitmap, 0)
    val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    return try {
        val visionText = recognizer.process(image).await()
        val parsed = parseFieldsNoRegex(visionText)
        parsed
    } catch (e: Exception) {
        Log.e("LicenseParse", "Error parsing license data", e)
        CambodianDrivingLicense(rawText = "Error: ${e.message}")
    }
}



/* ----------------------------- UI (unchanged) ----------------------------- */

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

    // Replace this with your asset/drawable
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
                if (isLoading) return@Button
                isLoading = true
                recognizedText = "Processing..."
                scope.launch {
                    try {
                        val result = extractCambodianLicenseData(context, imageToRecognize)
                        recognizedText = """
                            Surname (EN): ${result.surnameEnglish ?: "N/A"}
                            Given Name (EN): ${result.givenNameEnglish ?: "N/A"}
                            Sex: ${result.sex ?: "N/A"}
                            Nationality: ${result.nationality ?: "N/A"}
                            ID No: ${result.idNumber ?: "N/A"}
                            DOB: ${result.dateOfBirth ?: "N/A"}
                            Issue: ${result.dateOfIssue ?: "N/A"}
                            Expiry: ${result.dateOfExpiry ?: "N/A"}
                            Category: ${result.category ?: "N/A"}
                            Card Code: ${result.cardCode ?: "N/A"}
                            Address: ${result.address ?: "N/A"}
                            Place of Birth: ${result.placeOfBirth ?: "N/A"}
                            
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
        ) { Text("Recognize Text from Static Image") }

        Spacer(modifier = Modifier.height(16.dp))

        if (isLoading) {
            CircularProgressIndicator()
            Spacer(modifier = Modifier.height(16.dp))
        }

        Text(text = "Recognized Text:", style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(8.dp))
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(25.dp)
        ) {
            Text(text = recognizedText, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Preview(showBackground = true)
@Composable
fun TextRecognitionScreenPreview() {
    TextrecognationmlkitTheme { TextRecognitionScreen() }
}
