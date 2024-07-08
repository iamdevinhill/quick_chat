# Import necessary libraries and modules
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware

# Load pre-trained model and tokenizer from Hugging Face Transformers library
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

# Define special token ids
pad_token_id = tokenizer.eos_token_id  # Use eos_token_id for open-end generation

# Initialize FastAPI instance
app = FastAPI()

# CORS (Cross-Origin Resource Sharing) middleware setup to allow requests from all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development, restrict in production)
    allow_methods=["POST"],  # Allow only POST requests
    allow_headers=["*"],  # Allow all headers
)

# Define an endpoint for handling chat requests via POST method
@app.post("/chat/")
async def chat(input_data: dict):
    try:
        # Extract input_text from the request data
        input_text = input_data['input_text']

        # Tokenize the input text using the loaded tokenizer
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        # Generate attention mask to avoid attention on padding tokens
        attention_mask = input_ids.ne(pad_token_id).long()

        # Generate a response from the model based on the input text
        output = model.generate(input_ids, max_length=500, num_return_sequences=1,
                                no_repeat_ngram_size=2, pad_token_id=pad_token_id,
                                attention_mask=attention_mask)

        # Decode the generated output into text, skipping special tokens
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the input text (prompt) from the beginning of the generated text if present
        if generated_text.startswith(input_text):
            response = generated_text[len(input_text):].strip()
        else:
            response = generated_text.strip()

        # Return the generated response
        return {"response": response}

    except Exception as e:
        # Raise HTTPException with status code 422 if an error occurs
        raise HTTPException(status_code=422, detail=str(e))

# Start FastAPI server using uvicorn if the script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
