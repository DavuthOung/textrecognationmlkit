package com.example.text_recognation_ml_kit
import android.graphics.Rect
import com.google.mlkit.vision.text.Text
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.time.format.ResolverStyle

// ---------- small utilities (no regex) ----------
private const val MIN_OCR_CONF = 0.6f

fun collapseSpaces(s: String): String {
    val sb = StringBuilder(s.length)
    var seenSpace = false
    for (ch in s.trim()) {
        if (ch.isWhitespace()) {
            if (!seenSpace) sb.append(' ')
            seenSpace = true
        } else {
            sb.append(ch)
            seenSpace = false
        }
    }
    return sb.toString()
}

fun keepAZ09Space(s: String): String {
    val sb = StringBuilder(s.length)
    for (ch in s) {
        if (ch.isLetterOrDigit() || ch == ' ') sb.append(ch)
    }
    return sb.toString()
}

fun normLabel(s: String): String = keepAZ09Space(
    collapseSpaces(
        s.uppercase()
            .replace("&", " AND ")
            .replace("’", "'")
            .replace("`", "'")
            .replace(".", " ")
            .replace("-", " ")
            .replace("_", " ")
    )
).trim()

fun normValue(s: String): String = collapseSpaces(s)

// Jaccard + containment
fun labelSimilarity(a: String, b: String): Double {
    val A = normLabel(a)
    val B = normLabel(b)
    if (A.isEmpty() || B.isEmpty()) return 0.0
    if (A == B) return 1.0
    if ((A.length >= 4 && B.contains(A)) || (B.length >= 4 && A.contains(B))) return 0.92
    val asSet = A.split(' ').filter { it.isNotBlank() }.toSet()
    val bsSet = B.split(' ').filter { it.isNotBlank() }.toSet()
    val inter = asSet.intersect(bsSet).size.toDouble()
    val union = (asSet.size + bsSet.size - inter).toDouble().coerceAtLeast(1.0)
    return inter / union
}

// ---------- known labels per field ----------
val LABELS = mapOf(
    "name" to listOf("SURNAME AND NAME", "SURNAME & NAME", "NAME", "FULL NAME"),
    "sex" to listOf("SEX", "GENDER"),
    "nationality" to listOf("NATIONALITY"),
    "dob" to listOf("DATE OF BIRTH", "DOB"),
    "issue" to listOf("ISSUE DATE", "ISSUED DATE", "DATE OF ISSUE"),
    "expiry" to listOf("EXPIRY DATE", "VALID UNTIL", "EXPIRY"),
    "id" to listOf("ID", "ID NO", "ID NUMBER", "NID"),
    "category" to listOf("CATEGORY", "CATEGORIES", "CLASS"),
    "cardCode" to listOf("CARD CODE", "CARD NO", "CODE", "CARDCODE"),
    "address" to listOf("ADDRESS"),
    "pob" to listOf("PLACE OF BIRTH"),
    "special" to listOf("SPECIAL CONDITION", "SPECIAL CONDITIONS")
)

// ---------- date parsing (no regex) ----------
val dateFormatters: List<DateTimeFormatter> = listOf(
    "d/M/uuuu", "dd/MM/uuuu", "d-M-uuuu", "dd-MM-uuuu",
    "d.M.uuuu", "dd.MM.uuuu", "d/M/uu", "dd/MM/uu",
    "M/d/uuuu", "MM/dd/uuuu", "uuuu-MM-dd"
).map { DateTimeFormatter.ofPattern(it).withResolverStyle(ResolverStyle.SMART) }

fun tryParseDateIso(s: String): String? {
    val t = collapseSpaces(s)
    for (fmt in dateFormatters) {
        try {
            val d = LocalDate.parse(t, fmt)
            if (d.year in 1900..2100) return d.toString()
        } catch (_: Throwable) { /* try next */ }
    }
    // last resort: split by -, /, .
    val t2 = t.replace('-', ' ').replace('/', ' ').replace('.', ' ')
    val parts = t2.split(' ').filter { it.isNotBlank() }
    if (parts.size == 3) {
        val d = parts[0].toIntOrNull()
        val m = parts[1].toIntOrNull()
        var y = parts[2].toIntOrNull()
        if (d != null && m != null && y != null) {
            if (y < 100) y = if (y >= 50) 1900 + y else 2000 + y
            if (m in 1..12 && d in 1..31 && y in 1900..2100) {
                return LocalDate.of(y, m, d).toString()
            }
        }
    }
    return null
}

fun isLikelyCategoryToken(t: String): Boolean {
    val x = t.uppercase()
    if (x == "AUTO") return false
    if (x == "A" || x == "B" || x == "C" || x == "D" || x == "E" || x == "M") return true
    if (x.length in 2..3 && x[0] == 'A' && x[1].isDigit()) return true // A1, A2
    return false
}

fun looksLikeCardCodeToken(t: String): Boolean {
    // Simple heuristic: has at least one dot, at least one digit, and a letter
    var hasDot = false; var hasDigit = false; var hasAlpha = false
    for (ch in t) {
        if (ch == '.') hasDot = true
        if (ch.isDigit()) hasDigit = true
        if (ch.isLetter()) hasAlpha = true
    }
    return hasDot && hasDigit && hasAlpha
}

fun collapseDelimsToSpaces(s: String): String =
    s.replace(',', ' ').replace('/', ' ').replace('|', ' ').replace(':', ' ')

// ---------- OCR tokenization ----------
data class OcrToken(
    val text: String,
    val box: Rect,
    val cx: Float,
    val cy: Float,
    val lineId: Int,
    val blockId: Int,
)

fun tokensFrom(visionText: Text, minConfidence: Float = MIN_OCR_CONF): List<OcrToken> {
    val tokens = mutableListOf<OcrToken>()
    var lineCounter = 0
    var blockCounter = 0

    fun confOrDefault(getter: () -> Float): Float =
        try { getter().let { if (it.isNaN()) 1f else it } } catch (_: Throwable) { 1f }

    for (b in visionText.textBlocks) {
        val bId = blockCounter++
        for (l in b.lines) {
            val lineConf = confOrDefault { l.confidence }
            if (lineConf < minConfidence) continue

            val lId = lineCounter++
            for (e in l.elements) {
                val elConf = confOrDefault { e.confidence }
                if (elConf < minConfidence) continue

                val r = e.boundingBox ?: continue
                tokens += OcrToken(
                    text = e.text,
                    box = r,
                    cx = (r.left + r.right) / 2f,
                    cy = (r.top + r.bottom) / 2f,
                    lineId = lId,
                    blockId = bId
                )
            }
        }
    }
    return tokens.sortedWith(compareBy<OcrToken> { it.lineId }.thenBy { it.cx })
}


fun lineBands(tokens: List<OcrToken>): Map<Int, Pair<Float, Float>> =
    tokens.groupBy { it.lineId }.mapValues { (_, ts) ->
        val top = ts.minOf { it.box.top }.toFloat()
        val bot = ts.maxOf { it.box.bottom }.toFloat()
        top to bot
    }

fun bestLabelKeyFor(text: String, minScore: Double = 0.84): String? {
    var bestKey: String? = null
    var bestScore = 0.0
    for ((key, variants) in LABELS) {
        for (v in variants) {
            val s = labelSimilarity(text, v)
            if (s > bestScore) { bestScore = s; bestKey = key }
        }
    }
    return if (bestScore >= minScore) bestKey else null
}

fun collectRightOrBelow(
    labelTok: OcrToken,
    tokens: List<OcrToken>,
    bands: Map<Int, Pair<Float, Float>>,
    xGapMinPx: Int = 6
): String {
    // same line, to the right
    val right = tokens.filter { it.lineId == labelTok.lineId && it.cx > (labelTok.box.right + xGapMinPx) }
        .sortedBy { it.cx }
        .joinToString(" ") { it.text }
        .let { normValue(collapseSpaces(it)) }
    if (right.isNotBlank()) return right

    // next line
    val nextLineId = tokens.map { it.lineId }.filter { it > labelTok.lineId }.minOrNull()
    if (nextLineId != null) {
        val below = tokens.filter { it.lineId == nextLineId }.sortedBy { it.cx }
            .joinToString(" ") { it.text }
            .let { normValue(collapseSpaces(it)) }
        if (below.isNotBlank()) return below
    }
    return ""
}

fun splitName(full: String): Pair<String?, String?> {
    val toks = collapseSpaces(full).split(' ').filter { it.isNotBlank() }
    return when {
        toks.isEmpty() -> null to null
        toks.size == 1 -> null to toks.first()
        else -> toks.first() to toks.drop(1).joinToString(" ")
    }
}
