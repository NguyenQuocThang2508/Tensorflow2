from transformers import pipeline

# 1. Phân tích cảm xúc (Sentiment Analysis)
def sentiment_analysis(text):
    print("\n==== Phân tích cảm xúc ====")
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)
    for res in result:
        print(f"Text: {text}")
        print(f"Sentiment: {res['label']} (Score: {res['score']:.2f})")
    print("===========================\n")


# 2. Trả lời câu hỏi (Question Answering)
def question_answering(question, context):
    print("\n==== Trả lời câu hỏi ====")
    qa_pipeline = pipeline("question-answering")
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']} (Score: {result['score']:.2f})")
    print("=========================\n")


# 3. Tóm tắt văn bản (Summarization)
def summarize_text(text):
    print("\n==== Tóm tắt văn bản ====")
    summarizer = pipeline("summarization")
    result = summarizer(text, max_length=50, min_length=25, do_sample=False)
    for res in result:
        print(f"Original Text: {text}")
        print(f"Summary: {res['summary_text']}")
    print("=========================\n")


# 4. Sinh văn bản (Text Generation)
def text_generation(prompt, max_length=50):
    print("\n==== Sinh văn bản ====")
    generator = pipeline("text-generation", model="gpt2")
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    for res in result:
        print(f"Prompt: {prompt}")
        print(f"Generated Text: {res['generated_text']}")
    print("======================\n")


# === Chương trình chính ===
if __name__ == "__main__":
    # === Ví dụ 1: Phân tích cảm xúc ===
    sentiment_analysis("I am so happy with the progress we made!")
    sentiment_analysis("This product is terrible and I hate using it.")

    # === Ví dụ 2: Trả lời câu hỏi ===
    context = """
    Hugging Face Transformers is an open-source library that provides pre-trained models for various 
    natural language processing tasks. It supports tasks such as text classification, summarization, and question answering.
    """
    question = "What tasks does Hugging Face Transformers support?"
    question_answering(question, context)

    # === Ví dụ 3: Tóm tắt văn bản ===
    long_text = """
    Hugging Face Transformers is an open-source library for natural language processing tasks. 
    It allows developers to use pre-trained models for tasks like text generation, classification, 
    summarization, and translation. Hugging Face supports both PyTorch and TensorFlow, making it 
    versatile for researchers and developers. The library also provides tools for fine-tuning models on custom datasets.
    """
    summarize_text(long_text)

    # === Ví dụ 4: Sinh văn bản ===
    text_generation("Once upon a time in a magical forest, there lived a group of friendly animals.")
    text_generation("The future of artificial intelligence is", max_length=50)
