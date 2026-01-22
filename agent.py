from memory.qdrant_store import retrieve_facts
from transformers import pipeline

def healthcare_agent(patient_data: dict):
    """
    Healthcare assistant function that generates a response based on patient symptoms,
    medical history, and demographic information. Uses Hugging Face Transformers for text generation.
    
    Parameters:
        patient_data (dict): Dictionary containing patient info:
            - symptoms (list of str)
            - age (int)
            - gender (str)

    Returns:
        str: Generated healthcare assistant response with disclaimer.
    """

    # Extract patient data safely
    symptoms = patient_data.get("symptoms", [])
    age = patient_data.get("age", "Unknown")
    gender = patient_data.get("gender", "Unknown")

    # Build query from symptoms
    query = " ".join(symptoms) if symptoms else "No symptoms provided"

    # Retrieve relevant medical history facts
    memory_facts = retrieve_facts(query)
    memory_facts = list(set(memory_facts))  # Deduplicate facts

    # Format medical history text
    if memory_facts:
        facts_text = "\n".join(f"- {fact}" for fact in memory_facts)
        history_section = f"\nRelevant medical history:\n{facts_text}\n"
    else:
        facts_text = "No prior medical history."
        history_section = "\nNo prior medical history found.\n"

    # Initialize language model pipeline
    llm = pipeline(
        "text-generation",
        model="distilgpt2",
        tokenizer="distilgpt2"
    )

    # Construct prompt for the model
    prompt = f"""
    You are a healthcare assistant. Based on these facts:
    {facts_text}

    Provide a short, clear explanation of how these facts might relate to the user's question: {query}.
    """

    # Generate response (avoid echoing prompt)
    generated = llm(
        prompt,
        max_new_tokens=256,   # Use only one length parameter
        do_sample=True,
        truncation=True       # Explicit truncation to avoid warnings
    )[0]["generated_text"]

    # Remove prompt echo if present
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()

    # Build final response
    response = (
        f"Based on your symptoms:\n"
        f"{history_section}"
        f"{generated}\n\n"
        f"Patient info:\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n\n"
        "⚠️ This output is for informational purposes only. "
        "It is not a medical diagnosis. Please consult a qualified healthcare professional."
    )

    return response

      
