# SWYNEX Model or API Integration – Task 2

## AI-Based Customer Support Ticket Classifier

This project is Task 2 of my SWYNEX Technologies internship. It continues the AI problem defined in Task 1 by implementing a working machine learning prototype.

## Project Objective

The objective is to automatically classify customer support messages into the correct support category.

The system supports five categories:

- Payment Issue
- Delivery Issue
- Product Issue
- Refund / Return
- Other

## Model Integration

The prototype is built using Python and the Scikit-learn machine learning library.

The classification pipeline uses:

1. **TF-IDF Vectorizer** – Converts customer text into numerical features.
2. **Logistic Regression** – Predicts the most suitable support category.
3. **Scikit-learn Pipeline** – Combines text processing and classification into one workflow.

No external API or secret API key is required.

## Dataset

The prototype uses a small labeled dataset containing 50 customer support messages.

Each record contains:

- `customer_message`
- `category`

The dataset contains 10 sample messages for each of the five categories.

## How It Works

The workflow is:

Customer Message → TF-IDF Vectorization → Logistic Regression Model → Predicted Category

The model is trained when the Python program starts. The user can then enter a customer support message and receive a predicted category and confidence score.

## Example Inputs and Outputs

### Example 1

Input:

`My delivery has not arrived yet`

Output:

`Delivery Issue`

### Example 2

Input:

`I want a refund for my order`

Output:

`Refund / Return`

### Example 3

Input:

`My product is broken`

Output:

`Product Issue`

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
