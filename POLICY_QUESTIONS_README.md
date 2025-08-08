# Policy Questions API Enhancement

This enhancement adds a specialized endpoint for answering National Parivar Mediclaim Plus Policy questions with structured responses and confidence scoring.

## New Features

### 1. Enhanced LLM Service

- Added `answer_policy_questions()` method to `LLMService` class
- Specialized prompt engineering for insurance policy questions
- Structured JSON response with confidence scores and inference explanations

### 2. New API Endpoint

- **Endpoint**: `POST /policy-questions/`
- **Purpose**: Answer specific insurance policy questions with detailed analysis
- **Response Format**: Structured answers with confidence scores and reasoning

### 3. Request/Response Models

- `PolicyQuestionsRequest`: Input model for policy questions
- `PolicyQuestionsResponse`: Structured output with answers, confidence scores, and inferences

## API Usage

### Endpoint Details

```
POST /policy-questions/
Content-Type: application/json
```

### Request Format

```json
{
  "questions": [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "..."
  ],
  "use_context": true
}
```

### Response Format

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date...",
    "There is a waiting period of thirty-six (36) months of continuous coverage...",
    "..."
  ],
  "confidence_scores": [0.95, 0.87, 0.92, ...],
  "inferences": [
    "Based on standard insurance policy grace period clauses...",
    "Derived from typical pre-existing disease waiting period provisions...",
    "..."
  ],
  "average_confidence": 0.91,
  "processing_time": 2.34
}
```

### Key Response Fields

- **answers**: List of detailed, factual answers to each question
- **confidence_scores**: Float values (0.0-1.0) indicating answer confidence
- **inferences**: Explanations of how each answer was derived
- **average_confidence**: Overall confidence across all answers
- **processing_time**: Time taken to process the request

## Testing

### 1. Python Test Script

```bash
python test_policy_api.py
```

### 2. cURL Test

```bash
./test_policy_curl.sh
```

### 3. Direct LLM Service Test

```bash
python test_policy_questions.py
```

## Example Questions Supported

1. Grace period for premium payment
2. Waiting periods for various conditions
3. Coverage for maternity, organ donation, AYUSH treatments
4. No Claim Discount details
5. Hospital definitions and requirements
6. Sub-limits and coverage restrictions

## Configuration

The endpoint uses the existing LLM service configuration:

- **Environment Variables**: `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `DEFAULT_MODEL`
- **Context Integration**: Optionally uses vector database context for enhanced accuracy
- **Temperature**: Low temperature (0.1) for consistent, factual responses

## Benefits

1. **Structured Output**: Consistent JSON format for easy integration
2. **Confidence Scoring**: Quantified reliability metrics for each answer
3. **Inference Tracking**: Transparency in how answers were derived
4. **Context Awareness**: Integration with existing document vector database
5. **Error Handling**: Robust fallback mechanisms for parsing errors

## Integration with Existing System

- Uses existing `LLMService` infrastructure
- Integrates with ChromaDB for context retrieval
- Maintains consistent logging and error handling
- Compatible with existing FastAPI architecture
- Follows established response patterns

This enhancement maintains backward compatibility while adding specialized functionality for insurance policy question answering.
