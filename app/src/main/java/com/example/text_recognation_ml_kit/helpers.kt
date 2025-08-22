package com.example.text_recognation_ml_kit

import android.util.Log
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.YearMonth
import java.time.format.DateTimeFormatter
import java.time.format.DateTimeParseException
import java.util.Locale

object GeneralDateConverter {

	/**
	 * Converts a date string from one format to another.
	 *
	 * @param inputDateString The date string to convert.
	 * @param inputPattern The pattern of the inputDateString (e.g., "yyMMdd", "yyyy-MM-dd HH:mm:ss").
	 * @param outputPattern The desired pattern for the output date string (e.g., "dd MMM yyyy", "MM/dd/yyyy").
	 * @param inputLocale The locale used for parsing the input string (important if input has month names, etc.). Defaults to Locale.US.
	 * @param outputLocale The locale used for formatting the output string (important for output month names, etc.). Defaults to Locale.US.
	 * @return The converted date string in the outputPattern format, or null if conversion fails.
	 */
	fun convertDateString(
		inputDateString: String?,
		inputPattern: String,
		outputPattern: String,
		inputLocale: Locale = Locale.getDefault(),
		outputLocale: Locale = Locale.getDefault()
	): String? {
		if (inputDateString.isNullOrBlank()) {
			return "N/A"
		}

		val inputFormatter = try {
			DateTimeFormatter.ofPattern(inputPattern, inputLocale)
		} catch (e: IllegalArgumentException) {
			// Log.e("GeneralDateConverter", "Invalid input pattern: $inputPattern", e)
			return "N/A"
		}

		val outputFormatter = try {
			DateTimeFormatter.ofPattern(outputPattern, outputLocale)
		} catch (e: IllegalArgumentException) {
			// Log.e("GeneralDateConverter", "Invalid output pattern: $outputPattern", e)
			return "N/A"
		}

		try {
			// Try parsing as LocalDate first, as it's common.
			// If the inputPattern contains time components, this will fail, and we'll try LocalDateTime.
			// For MRZ YYMMDD, LocalDate is sufficient.
			val temporalAccessor = try {
				LocalDate.parse(inputDateString, inputFormatter)
			} catch (e: DateTimeParseException) {
				try {
					// If LocalDate fails, try LocalDateTime if the pattern suggests time
					if (inputPattern.contains("H") || inputPattern.contains("h") ||
						inputPattern.contains("m") || inputPattern.contains("s")
					) {
						LocalDateTime.parse(inputDateString, inputFormatter)
					} else if (inputPattern.length <= "yyyyMM".length && (inputPattern.contains("yyyy") && inputPattern.contains(
							"MM"
						) && !inputPattern.contains("dd"))
					) {
						// Handle YearMonth if only year and month are present
						YearMonth.parse(inputDateString, inputFormatter)
					} else {
						throw e // Re-throw if it's not a time-related pattern or YearMonth
					}
				} catch (e2: DateTimeParseException) {
					// Log.e("GeneralDateConverter", "Failed to parse date string '$inputDateString' with pattern '$inputPattern'", e2)
					return "N/A"
				}
			}
			return outputFormatter.format(temporalAccessor)

		} catch (e: Exception) { // Catch any other unexpected errors during formatting
			// Log.e("GeneralDateConverter", "Error during date conversion: ${e.message}", e)
			return "N/A"
		}
	}
}

// Data class to hold parsed information
data class ParsedMrzInfo(
	val documentType: DocumentType = DocumentType.UNKNOWN,
	val documentCode: String = "", // e.g., "P<", "ID", "IV"
	val issuingCountry: String = "", // ISO 3166-1 alpha-3 code
	val surname: String = "",
	val givenNames: String = "",
	val documentNumber: String = "",
	val documentNumberCheckDigit: Char? = null,
	val nationality: String = "", // ISO 3166-1 alpha-3 code
	val dateOfBirth: String = "", // YYMMDD
	val dateOfBirthCheckDigit: Char? = null,
	val sex: Sex = Sex.UNSPECIFIED,
	val dateOfExpiry: String? = "", // YYMMDD
	val dateOfExpiryCheckDigit: Char? = null,
	val optionalData1: String = "", // For passport, this is personal number or other nat. data
	val optionalData1CheckDigit: Char? = null, // For passport TD3
	val overallCheckDigit: Char? = null,
	val rawMrzLines: List<String> = emptyList(),
	val parsingErrors: MutableList<String> = mutableListOf(),
	var documentNumberValid: Boolean? = null,
	var dateOfBirthValid: Boolean? = null,
	var dateOfExpiryValid: Boolean? = null,
	var optionalData1Valid: Boolean? = null, // For passport TD3
	var overallValid: Boolean? = null
)

enum class DocumentType {
	PASSPORT_TD3, // Typically 2 lines, 44 chars each
	ID_CARD_TD1,  // Typically 3 lines, 30 chars each
	ID_CARD_TD2,  // Typically 2 lines, 36 chars each (can be similar to TD3 parsing)
	UNKNOWN
}

enum class Sex {
	MALE, FEMALE, UNSPECIFIED
}

object MrzParser {

	private const val TAG = "MrzParser"

	fun parse(mrzString: String): ParsedMrzInfo {
		val rawLines = mrzString.trim().split("\n")
			.map { it.trim().replace(" ", "<") } // Replace spaces with fillers early
		if (rawLines.isEmpty()) {
			return ParsedMrzInfo(parsingErrors = mutableListOf("MRZ string is empty"))
		}

		// Basic type detection based on line count and length (simplification)
		val type: DocumentType
		val line1 = rawLines[0]
		val line2 = if (rawLines.size > 1) rawLines[1] else ""
		val line3 = if (rawLines.size > 2) rawLines[2] else ""

		// Simplified type detection - a robust solution would look at Document Code
		if (rawLines.size == 2 && line1.length >= 44 && line2.length >= 44) {
			type = DocumentType.PASSPORT_TD3 // Could also be MRVA
		} else if (rawLines.size == 3 && line1.length >= 30 && line2.length >= 30 && line3.length >= 30) {
			type = DocumentType.ID_CARD_TD1
		} else if (rawLines.size == 2 && line1.length >= 36 && line2.length >= 36) {
			type = DocumentType.ID_CARD_TD2 // or MRVB
		} else {
			Log.w(
				TAG,
				"Unknown MRZ format. Lines: ${rawLines.size}, Lengths: ${rawLines.map { it.length }}"
			)
			// Attempt generic parsing or return error
			type = when (rawLines.size) {
				2 -> {
					DocumentType.PASSPORT_TD3 // Default to passport-like if 2 lines
				}
				3 -> {
					DocumentType.ID_CARD_TD1   // Default to ID-like if 3 lines
				}
				else -> {
					return ParsedMrzInfo(
						rawMrzLines = rawLines,
						parsingErrors = mutableListOf("Cannot determine MRZ type reliably.")
					)
				}
			}

		}

		Log.d(TAG, "Attempting to parse as: $type")

		return when (type) {
			DocumentType.PASSPORT_TD3 -> parseTd3Passport(rawLines, type)
			DocumentType.ID_CARD_TD1 -> parseTd1IdCard(rawLines, type)
			DocumentType.ID_CARD_TD2 -> parseTd2IdCard(
				rawLines,
				type
			)
			else -> ParsedMrzInfo(
				rawMrzLines = rawLines,
				parsingErrors = mutableListOf("Unsupported MRZ type for parsing: $type")
			)
		}
	}

	// Inside MrzParser object
	private fun parseTd3Passport(lines: List<String>, docType: DocumentType): ParsedMrzInfo {
		val line1 = lines[0].padEnd(44, '<')
		val line2 = lines[1].padEnd(44, '<')
		val errors = mutableListOf<String>()

		val documentCode = line1.substringSafe(0, 2)
		val issuingCountry = line1.substringSafe(2, 5).trim('<')
		val names = parseNameField(line1.substringSafe(5, 44))

		// Line 2 parsing
		val docNumRaw = line2.substringSafe(0, 9).trim('<') // Positions 1-9
		val docNumCheckChar = line2.getOrNullSafe(9)        // Position 10
		val nationality = line2.substringSafe(10, 13).trim('<') // Positions 11-13
		val dobRaw = line2.substringSafe(13, 19)            // Positions 14-19
		val dobCheckChar = line2.getOrNullSafe(19)          // Position 20
		val sex = parseSex(line2.getOrNullSafe(20))         // Position 21
		val expiryRaw = line2.substringSafe(21, 27)         // Positions 22-27
		val expiryCheckChar = line2.getOrNullSafe(27)       // Position 28
		val optional1Raw = line2.substringSafe(28, 42).trim('<') // Positions 29-42
		val optional1CheckChar = line2.getOrNullSafe(42)    // Position 43
		val overallCheckChar = line2.getOrNullSafe(43)      // Position 44

		val info = ParsedMrzInfo(
			documentType = docType,
			documentCode = documentCode,
			issuingCountry = issuingCountry,
			surname = names.first,
			givenNames = names.second,
			documentNumber = docNumRaw,
			documentNumberCheckDigit = docNumCheckChar,
			nationality = nationality,
			dateOfBirth = dobRaw,
			dateOfBirthCheckDigit = dobCheckChar,
			sex = sex,
			dateOfExpiry = GeneralDateConverter.convertDateString(expiryRaw, "yyyy-MM-dd", "dd/MM/yyyy"),
			dateOfExpiryCheckDigit = expiryCheckChar,
			optionalData1 = optional1Raw,
			optionalData1CheckDigit = optional1CheckChar,
			overallCheckDigit = overallCheckChar,
			rawMrzLines = lines,
			parsingErrors = errors
		)

		// Validate individual Check Digits
		info.documentNumberValid = validate(docNumRaw, docNumCheckChar, errors, "Document Number")
		info.dateOfBirthValid = validate(dobRaw, dobCheckChar, errors, "Date of Birth")
		info.dateOfExpiryValid = validate(expiryRaw, expiryCheckChar, errors, "Date of Expiry")

		// Only validate optional data check digit if optional data itself or its check digit seems present
		// and is not just a filler.
		if (optional1Raw.isNotEmpty() || (optional1CheckChar != null && optional1CheckChar != '<')) {
			info.optionalData1Valid =
				validate(optional1Raw, optional1CheckChar, errors, "Optional Data 1")
		} else {
			// If optional data is all fillers, and its check digit is a filler, it's considered valid/not applicable
			if (optional1Raw.all { it == '<' }) {
				info.optionalData1Valid = true // Or null if you prefer to signify N/A
			} else if (optional1Raw.isNotEmpty()) { // Optional data is present but its check digit is missing
				info.optionalData1Valid =
					validate(optional1Raw, optional1CheckChar, errors, "Optional Data 1")
			}
		}

		// --- CORRECTED Composite String for TD3 Overall Check Digit Validation ---
		val stringToValidateOverall =
			line2.substringSafe(0, 10) +  // Document number (9 chars) + its check digit (1 char)
					line2.substringSafe(
						13,
						20
					) + // Date of birth (6 chars) + its check digit (1 char)
					line2.substringSafe(
						21,
						28
					) + // Date of expiry (6 chars) + its check digit (1 char)
					line2.substringSafe(
						28,
						43
					)   // Optional data 1 (14 chars) + its check digit (1 char)
		// This concatenation results in a 10 + 7 + 7 + 15 = 39 character string

		// The overall check digit is at position 44 of line 2 (index 43)
		info.overallValid =
			validate(stringToValidateOverall, overallCheckChar, errors, "Overall TD3")

		return info
	}

	// Inside MrzParser object
	private fun parseTd1IdCard(lines: List<String>, docType: DocumentType): ParsedMrzInfo {
		val line1 = lines[0].padEnd(30, '<')
		val line2 = lines[1].padEnd(30, '<')
		val line3 = lines[2].padEnd(30, '<')
		val errors = mutableListOf<String>()

		val documentCode = line1.substringSafe(0, 2)
		val issuingCountry = line1.substringSafe(2, 5).trim('<')

		// --- New parsing for Line 1 Document Number and its optional parts ---
		val line1FieldData = line1.substringSafe(5, 29) // Data from pos 6 to 29 (24 chars)
		val line1FieldCheckDigitChar = line1.getOrNullSafe(29) // Check digit at pos 30

		// Attempt to extract primary document number and its specific check digit if present within line1FieldData
		// This is heuristic, assuming the doc num + its CD is followed by fillers
		var primaryDocNum: String
		var primaryDocNumCheckDigit: Char? = null
		var line1OptionalData: String

		val lastNonFillerIndex = line1FieldData.indexOfLast { it != '<' }
		if (lastNonFillerIndex != -1) {
			val meaningfulContent = line1FieldData.substring(0, lastNonFillerIndex + 1)
			if (meaningfulContent.isNotEmpty() && meaningfulContent.last()
					.isDigitPossiblyFiller()
			) {
				// Assume last digit of meaningful content is the check digit for the part before it
				primaryDocNum = meaningfulContent.substringSafe(0, meaningfulContent.length - 1)
				primaryDocNumCheckDigit = meaningfulContent.last()

				// If primaryDocNum is now all fillers, it means meaningfulContent was just the check digit.
				// This logic might need refinement based on actual card structures.
				if (primaryDocNum.all { it == '<' }) primaryDocNum = ""


			} else {
				// No apparent embedded check digit, treat all meaningful content as document number
				primaryDocNum = meaningfulContent
			}
			// What remains of line1FieldData after primaryDocNum and its CD would be "optional data"
			// This is complex because optional data can also be part of the "document number" conceptually.
			// For now, let's assume if we found a primaryDocNumCD, line1OptionalData is what's not consumed.
			// This simplification might need adjustment based on ICAO's definition for specific cards.
			// The string `line1FieldData` itself is what `line1FieldCheckDigitChar` validates.
			line1OptionalData = line1FieldData.substring(meaningfulContent.length).trim('<')


		} else {
			// line1FieldData is all fillers
			primaryDocNum = ""
			line1OptionalData = ""
		}
		// --- End of new parsing for Line 1 ---

		val dobRaw = line2.substringSafe(0, 6)
		val dobCheck = line2.getOrNullSafe(6)
		val sex = parseSex(line2.getOrNullSafe(7))
		val expiryRaw = line2.substringSafe(8, 14)
		val expiryCheck = line2.getOrNullSafe(14)
		val nationality = line2.substringSafe(15, 18).trim('<')
		val overallCheck = line2.getOrNullSafe(29) // This is the overall MRZ check digit

		val names = parseNameField(line3.padEnd(30, '<'))
		Log.i("MRZ", "Parsed Expiry date: ${GeneralDateConverter.convertDateString(expiryRaw.trimEnd('<'), "yyyy-MM-dd", "dd/MM/yyyy")}")
		val info = ParsedMrzInfo(
			documentType = docType,
			documentCode = documentCode,
			issuingCountry = issuingCountry,
			surname = names.first,
			givenNames = names.second,
			documentNumber = primaryDocNum.trimEnd('<'), // Store the trimmed primary doc num
			documentNumberCheckDigit = if (primaryDocNumCheckDigit == '<') null else primaryDocNumCheckDigit, // Store its specific CD
			nationality = nationality,
			dateOfBirth = dobRaw.trimEnd('<'),
			dateOfBirthCheckDigit = dobCheck,
			sex = sex,
			dateOfExpiry = expiryRaw.trimEnd('<'),
			dateOfExpiryCheckDigit = expiryCheck,
			optionalData1 = line1OptionalData, // Optional data from line 1
			// Optional data 2 from line 2 can be stored in a different field or combined
			// optionalData2 = optionalL2Part,
			overallCheckDigit = overallCheck,
			rawMrzLines = lines,
			parsingErrors = errors
		)

		// Validate the *primary* document number against its *specific* check digit (if found)
		if (primaryDocNum.isNotEmpty() || primaryDocNumCheckDigit != null) {
			info.documentNumberValid =
				validate(primaryDocNum, primaryDocNumCheckDigit, errors, "Document Number")
		}


		// Validate the entire line1FieldData (pos 6-29) against line1FieldCheckDigitChar (pos 30)
		// This is the check digit for the "composite document number / optional data 1" field
		// Only if line1FieldCheckDigitChar is not '<'
		if (line1FieldCheckDigitChar != null && line1FieldCheckDigitChar != '<') {
			validate(line1FieldData, line1FieldCheckDigitChar, errors, "Line 1 Field (DocNum+Opt1)")
		} else if (line1FieldCheckDigitChar == null && line1FieldData.any { it != '<' }) {
			errors.add("Line 1 Field (DocNum+Opt1): Check digit missing for non-empty field.")
		}


		info.dateOfBirthValid = validate(dobRaw, dobCheck, errors, "Date of Birth (L2)")
		info.dateOfExpiryValid = validate(expiryRaw, expiryCheck, errors, "Date of Expiry (L2)")

		// Composite for TD1 overall MRZ check digit (at end of line 2)
		// This uses fixed length substrings from the padded lines.
		val compositeStringForOverallValidation =
			line1.substring(5, 30) +  // Document number field (24 chars) + its check digit (1 char)
					line2.substring(0, 7) +   // Date of birth (6 chars) + its check digit (1 char)
					line2.substring(8, 15) +  // Date of expiry (6 chars) + its check digit (1 char)
					line2.substring(18, 29)   // Optional data from line 2 (11 chars)
		// Total 25 + 7 + 7 + 11 = 50 characters

		info.overallValid =
			validate(compositeStringForOverallValidation, overallCheck, errors, "Overall TD1")

		return info
	}

	// Helper for Char.isDigit or if it's a filler that might have been a digit
	private fun Char.isDigitPossiblyFiller(): Boolean = this.isDigit() || this == '<'

	private fun parseTd2IdCard(lines: List<String>, docType: DocumentType): ParsedMrzInfo {
		// TD2 (2 lines, 36 chars) is similar to TD3 in field structure of line 2, but line 1 is different
		// Line 1: DocCode (2), IssuingCountry (3), Name (31)
		// Line 2: DocNum (9), CD (1), Nationality (3), DOB (6), CD (1), Sex (1), Expiry (6), CD (1), Optional (1), OverallCD (1) -> (this optional is often omitted or fixed length)
		val line1 = lines[0].padEnd(36, '<')
		val line2 = lines[1].padEnd(36, '<')
		val errors = mutableListOf<String>()

		val documentCode = line1.substringSafe(0, 2)
		val issuingCountry = line1.substringSafe(2, 5).trim('<')
		val names = parseNameField(line1.substringSafe(5, 36)) // Name fills rest of line 1

		val docNumRaw = line2.substringSafe(0, 9).trim('<')
		val docNumCheck = line2.getOrNullSafe(9)
		val nationality = line2.substringSafe(10, 13).trim('<')
		val dobRaw = line2.substringSafe(13, 19)
		val dobCheck = line2.getOrNullSafe(19)
		val sex = parseSex(line2.getOrNullSafe(20))
		val expiryRaw = line2.substringSafe(21, 27)
		val expiryCheck = line2.getOrNullSafe(27)
		// TD2 has a single character optional data field before overall check digit typically.
		val optional1Raw = line2.substringSafe(28, 35).trim('<') //ICAO TD2 spec has optional data here before overall check digit
		val overallCheck = line2.getOrNullSafe(35)


		val info = ParsedMrzInfo(
			documentType = docType,
			documentCode = documentCode,
			issuingCountry = issuingCountry,
			surname = names.first,
			givenNames = names.second,
			documentNumber = docNumRaw,
			documentNumberCheckDigit = docNumCheck,
			nationality = nationality,
			dateOfBirth = dobRaw,
			dateOfBirthCheckDigit = dobCheck,
			sex = sex,
			dateOfExpiry = expiryRaw,
			dateOfExpiryCheckDigit = expiryCheck,
			optionalData1 = optional1Raw, // Optional data for TD2
			// optionalData1CheckDigit = null, // TD2 typically doesn't have a separate check digit for this small optional field
			overallCheckDigit = overallCheck,
			rawMrzLines = lines,
			parsingErrors = errors
		)

		// Validate Check Digits
		info.documentNumberValid = validate(docNumRaw, docNumCheck, errors, "Document Number")
		info.dateOfBirthValid = validate(dobRaw, dobCheck, errors, "Date of Birth")
		info.dateOfExpiryValid = validate(expiryRaw, expiryCheck, errors, "Date of Expiry")
		// No separate check digit for TD2's short optional field usually.

		// Composite for TD2 overall check digit (Refer to ICAO 9303 Part 3 or similar)
		// Simplified:
		val compositeStringForOverallValidation = docNumRaw.padEnd(9, '<') +
				docNumCheck.toString().padEnd(1, '<') +
				dobRaw.padEnd(6, '<') +
				dobCheck.toString().padEnd(1, '<') +
				expiryRaw.padEnd(6, '<') +
				expiryCheck.toString().padEnd(1, '<') +
				optional1Raw.padEnd(7, '<') // TD2 optional field length

		info.overallValid =
			validate(compositeStringForOverallValidation, overallCheck, errors, "Overall TD2")

		return info
	}


	private fun parseNameField(nameField: String): Pair<String, String> {
		val parts = nameField.split("<<", limit = 2)
		val surname = parts.getOrNull(0)?.replace("<", " ")?.trim() ?: ""
		val givenNames = if (parts.size > 1) parts[1].replace("<", " ").trim() else ""
		return Pair(surname, givenNames)
	}

	private fun parseSex(sexChar: Char?): Sex {
		return when (sexChar) {
			'M' -> Sex.MALE
			'F' -> Sex.FEMALE
			else -> Sex.UNSPECIFIED
		}
	}

	private fun getCharValue(char: Char): Int {
		return when {
			char.isDigit() -> char.toString().toInt()
			char in 'A'..'Z' -> char - 'A' + 10
			char == '<' -> 0
			else -> {
				Log.w(TAG, "Invalid character in MRZ for check digit calculation: '$char'")
				0 // Or throw error
			}
		}
	}

	private fun calculateCheckDigit(data: String): Int {
		if (data.isEmpty()) return 0 // Or handle as error
		val weights = intArrayOf(7, 3, 1)
		var sum = 0
		for (i in data.indices) {
			sum += getCharValue(data[i]) * weights[i % 3]
		}
		return sum % 10
	}

	private fun validate(
		dataField: String,
		expectedCheckDigitChar: Char?,
		errors: MutableList<String>,
		fieldName: String
	): Boolean? {
		// If the field to be checked is all fillers, and the check digit is also a filler or null, it's N/A or valid by default.
		if (dataField.all { it == '<' } && (expectedCheckDigitChar == null || expectedCheckDigitChar == '<')) {
			return true // Or null for N/A
		}

		if (expectedCheckDigitChar == null) {
			// Only add error if dataField has actual content
			if (dataField.any { it != '<' }) {
				errors.add("$fieldName: Check digit character is missing entirely for non-empty field.")
			}
			return false // Cannot be valid if check digit is missing for non-empty field
		}

		if (expectedCheckDigitChar == '<') {
			// If check digit is '<', it often means "not applicable" or "no check performed".
			// Only consider it an error if the data field itself has non-filler content.
			if (dataField.any { it != '<' }) {
				errors.add("$fieldName: Check digit is '<' (not applicable) for non-empty field '$dataField'. Validation ambiguous.")
			}
			return null // Return null to signify "cannot definitively validate" or "N/A"
		}

		val expected = expectedCheckDigitChar.toString().toIntOrNull()
		if (expected == null) {
			errors.add("$fieldName: Expected check digit '$expectedCheckDigitChar' is not a valid number.")
			return false
		}

		// Pad the dataField to its ICAO specified length for this specific check digit calculation if necessary
		// This part is crucial and depends on the field. E.g., a date is 6 chars, doc num for TD1's own CD might be variable.
		// For now, calculateCheckDigit works on the dataField as passed.
		val calculated = calculateCheckDigit(dataField)
		val isValid = calculated == expected
		if (!isValid) {
			errors.add("$fieldName: Check digit mismatch. Expected $expected, calculated $calculated (for data '$dataField')")
		}
		return isValid
	}
	// Helper extensions
	private fun String.substringSafe(startIndex: Int, endIndex: Int): String {
		if (startIndex < 0 || endIndex > this.length || startIndex > endIndex) return ""
		return this.substring(startIndex, endIndex)
	}

	private fun String.getOrNullSafe(index: Int): Char? {
		return if (index >= 0 && index < this.length) this[index] else null
	}
}