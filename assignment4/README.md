# Assignment 4: Self-Critique Loop RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline with a self-critique loop for answering software engineering questions using a provided knowledge base.

---

## Files

- **index_kb.py**  
  Loads `self_critique_loop_dataset.json`, computes embeddings using Azure OpenAI, and indexes records into a FAISS vector database.

- **agentic_rag_simplified.py**  
  Implements the agentic RAG pipeline with four key nodes:
    - `retrieve_kb`: retrieves the top-5 relevant KB snippets for the user question
    - `generate_answer`: generates an initial answer, citing snippets by `doc_id`
    - `critique_answer`: critiques the answer, outputting `"COMPLETE"` or `"REFINE: ..."`
    - `refine_answer`: incorporates missing info if refinement is needed  
  The main script processes five sample questions, showing all steps.

- **requirements.txt**  
  Lists all required Python packages.

- **self_critique_loop_dataset.json**  
  The provided dataset of software best-practice KB entries.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Azure OpenAI Setup
Set these environment variables before running any script:

On Linux/Mac:
```bash
export AZURE_OPENAI_ENDPOINT="your_end_point"
export AZURE_OPENAI_API_KEY="your_actual_key"
```

## Usage

1. Index the Knowledge Base
Build the FAISS KB index:
```bash
python index_kb.py
```
This will create a local faiss_kb index file.

2. Run the Agentic RAG Pipeline
Process sample questions, retrieve context, generate, critique, and refine answers:
```bash
python agentic_rag_simplified.py
```
You can edit or add user questions in the sample_questions list in agentic_rag_simplified.py.

## Output

For each user question, the script:

- Prints the top-5 retrieved KB snippets (kb_hits)
- Shows the initial_answer (with [KBxxx] citations)
- Shows the critique_result (COMPLETE or REFINE: ...)
- If refinement is needed, outputs a refined_answer
- Prints the final answer as a JSON object:

```json
{
  "answer": "When considering error handling, it's important to follow well-defined patterns to ensure consistency and reliability in your application. Key aspects to consider include:\n\n1. **Categorization of Errors**: Differentiate between types of errors (e.g., critical, non-critical, user errors) to handle them appropriately. This helps in determining the severity and the necessary response for each error type.\n\n2. **Logging**: Implement robust logging mechanisms to capture error details for debugging and monitoring purposes. Ensure that logs are structured and include relevant context to facilitate troubleshooting.\n\n3. **User Feedback**: Provide clear and actionable feedback to users when errors occur, avoiding technical jargon. This enhances the user experience by helping users understand what went wrong and what they can do next.\n\n4. **Graceful Degradation**: Ensure that your application can continue to function, even if certain features fail. This might involve providing fallback options or alternative workflows to maintain usability.\n\n5. **Recovery Strategies**: Develop strategies for error recovery, such as retry mechanisms or alternative actions that users can take. This can help mitigate the impact of errors and improve overall user satisfaction.\n\n6. **Testing**: Regularly test your error handling mechanisms to ensure they work as intended under various scenarios. This includes simulating different types of errors to validate that your application responds appropriately.\n\nBy adhering to these principles, you can create a more resilient and user-friendly application, ultimately leading to a better overall experience for your users. [KB009]"
}
```

