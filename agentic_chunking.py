# import os
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# # Load environment variables
# load_dotenv()

# # Ensure you have HUGGINGFACEHUB_API_TOKEN in your .env file

# # 1. Initialize the Hugging Face Endpoint
# # We use Llama-3-8B-Instruct as it handles structured prompting well.
# llm_endpoint = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation",
#     temperature=0.1,  # Low temperature for strict rule-following
#     max_new_tokens=512,
# )

# # 2. Wrap it in ChatHuggingFace to get clean chat message formatting
# llm = ChatHuggingFace(llm=llm_endpoint)

# # Tesla text to chunk
# tesla_text = """Tesla's Q3 Results
# Tesla reported record revenue of $25.2B in Q3 2024.
# The company exceeded analyst expectations by 15%.
# Revenue growth was driven by strong vehicle deliveries.

# Model Y Performance  
# The Model Y became the best-selling vehicle globally, with 350,000 units sold.
# Customer satisfaction ratings reached an all-time high of 96%.
# Model Y now represents 60% of Tesla's total vehicle sales.

# Production Challenges
# Supply chain issues caused a 12% increase in production costs.
# Tesla is working to diversify its supplier base.
# New manufacturing techniques are being implemented to reduce costs."""

# # 3. Use ChatPromptTemplate to format the prompt cleanly for the chat model
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "You are a text chunking expert. Your task is to split the user's text into logical chunks."),
#     ("user", """Rules:
# - Each chunk should be around 200 characters or less
# - Split at natural topic boundaries
# - Keep related information together
# - Put "<<<SPLIT>>>" between chunks

# Text:
# {text}

# Return the text with <<<SPLIT>>> markers where you want to split. Do not include any conversational filler or introductions, just the marked text.""")
# ])

# # Format the final prompt payload
# final_prompt = prompt_template.format_messages(text=tesla_text)

# # Get AI response
# print("🤖 Asking Hugging Face AI to chunk the text...")
# response = llm.invoke(final_prompt)
# marked_text = response.content

# # Split the text at the markers
# chunks = marked_text.split("<<<SPLIT>>>")

# # Clean up the chunks (remove extra whitespace)
# clean_chunks = []
# for chunk in chunks:
#     cleaned = chunk.strip()
#     if cleaned:  # Only keep non-empty chunks
#         clean_chunks.append(cleaned)

# # Show results
# print("\n🎯 AGENTIC CHUNKING RESULTS:")
# print("=" * 50)

# for i, chunk in enumerate(clean_chunks, 1):
#     print(f"Chunk {i}: ({len(chunk)} chars)")
#     print(f'"{chunk}"')
#     print()
