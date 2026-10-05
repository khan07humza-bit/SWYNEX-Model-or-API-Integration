import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv("customer_support_tickets.csv")

# Input messages and target categories
X = data["customer_message"]
y = data["category"]

# Create a text classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X, y)

print("=" * 55)
print("      AI CUSTOMER SUPPORT TICKET CLASSIFIER")
print("=" * 55)
print("\nModel trained successfully!")
print("Categories:")
print("- Payment Issue")
print("- Delivery Issue")
print("- Product Issue")
print("- Refund / Return")
print("- Other")

# Example predictions
example_messages = [
    "My package has not arrived",
    "I was charged twice",
    "The product is broken",
    "I want my money back",
    "I forgot my password"
]

print("\nExample Predictions:")
for message in example_messages:
    prediction = model.predict([message])[0]
    print(f"{message} -> {prediction}")

# Interactive prediction
print("\n" + "=" * 55)
print("Enter a customer message to classify it.")
print("Type 'exit' to close the program.")

while True:
    message = input("\nCustomer Message: ")

    if message.lower() == "exit":
        print("Classifier closed.")
        break

    prediction = model.predict([message])[0]
    confidence = model.predict_proba([message]).max() * 100

    print(f"Predicted Category: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
