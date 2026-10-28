# Label Engineering for E-commerce Order Support

## Overview

This project demonstrates **label engineering**: transforming raw business text into structured examples that can be used in machine-learning workflows. The original customer-support scenario has been changed to an **e-commerce order support** scenario.

The notebook uses a small illustrative dataset with customer messages about order status, order problems, payments, deliveries, returns, and refunds. It also includes order IDs, tracking codes, and product SKUs for sequence-tagging examples.

> **Important:** This is a teaching dataset. Its size and simplicity are not sufficient for production training or reliable performance evaluation.

## Learning goals

By working through the notebook, you will learn to:

- standardize raw issue types into consistent class labels;
- map class labels to numeric IDs;
- build instruction-response examples for supervised fine-tuning;
- format prompts and target labels;
- tag entity mentions at token level using the BIO convention;
- export processed datasets as CSV and JSON Lines (JSONL).

## Scenario and labels

The notebook's issue types are mapped to these labels:

| Raw issue type | Engineered label | Meaning |
|---|---|---|
| `order_status` | `ORDER_STATUS` | Asking where an order is or requesting its status |
| `order_problem` | `ORDER_PROBLEM` | A general problem with an order, such as receiving the wrong item |
| `payment` | `PAYMENT_ISSUE` | A payment failure or duplicate charge |
| `delivery` | `DELIVERY_ISSUE` | A delayed, missing, or incorrectly tracked package |
| `return` | `RETURN_REQUEST` | Asking to return an item |
| `refund` | `REFUND_REQUEST` | Asking about a refund |

These categories are examples for the exercise. In a real project, the label definitions should be documented, mutually understandable, and applied consistently by the people or processes labeling the data.

## Notebook walkthrough

### 1. Create the raw dataset

A pandas DataFrame stores the customer message, raw `issue_type`, and example `support_response`. These represent the source business data before label engineering.

### 2. Engineer multi-class labels

A dictionary maps raw issue types to standardized labels in `engineered_label`. A validation assertion checks that every issue type found in this dataset has a mapping. If a new issue type is introduced without a mapping, the check fails instead of silently leaving a missing label.

### 3. Convert labels into numeric IDs

The notebook sorts the unique engineered labels and creates two dictionaries:

- `label_to_id`: class label → integer ID
- `id_to_label`: integer ID → class label

The numeric `label_id` column is convenient for many model-training pipelines. The IDs are identifiers, not rankings: an ID of 5 does not mean a class is more important than a class with ID 1. Because IDs are assigned from sorted labels, changing the set of labels may change the mapping; save the mapping with the trained model and reuse it consistently.

### 4. Create instruction-response pairs

The `instruction_data` DataFrame contains:

- `instruction`: the task given to the model;
- `input`: the customer message;
- `response`: the expected class label.

This is a simple supervised fine-tuning style format: the model receives an instruction and input, and learns to produce the expected response.

### 5. Build prompt-target examples

`build_prompt()` combines the instruction, allowed label names, and customer message into a text prompt. The `target` is the expected label. Separating prompt from target makes the intended model input and desired output explicit.

### 6. Create sequence-tagging data

The entity examples identify a target value and its entity type, such as `ORD-1042` as an `ORDER_ID`, `TRK-78421` as a `TRACKING_CODE`, or `SKU-AX19` as a `PRODUCT_SKU`.

The notebook tokenizes text using whitespace splitting and creates a tag for each token:

- `B-TYPE`: the beginning (and, in these examples, the only token) of an entity;
- `I-TYPE`: a token inside a multi-token entity;
- `O`: a token outside the target entity.

**Implementation limitation:** The current `create_bio_tags()` function labels one exact entity value per sentence. It does not find multiple entities in one sentence, and it does not currently generate `I-` tags for multi-token entities. For example, the final example contains both an order ID and a tracking code, but the current row specifies only the order ID as its target. This is a simplified teaching implementation, not a complete named-entity recognition pipeline.

### 7. Export the datasets

When the notebook is run, it writes three files to the notebook's current working directory:

| Output file | Format | Contents |
|---|---|---|
| `ecommerce_multiclass_labels.csv` | CSV | Customer message, engineered label, numeric label ID |
| `ecommerce_instruction_response.jsonl` | JSONL | Instruction, input, response, prompt, target |
| `ecommerce_sequence_tags.jsonl` | JSONL | Text, tokens, BIO tags |

JSONL means **JSON Lines**: each line is a separate JSON object, which is useful for processing records one at a time.

## How to run

1. Open `4_Label-Engineering-Ecommerce.ipynb` in Jupyter Notebook, JupyterLab, or a compatible notebook environment.
2. Ensure Python and pandas are installed.
3. Run the notebook cells from top to bottom.
4. Inspect the printed DataFrames and token-level tags.
5. Confirm the three output files appear in the notebook's working directory.

## Practical checks before using real data

- Review class definitions and look for ambiguous or overlapping labels.
- Check for missing or unexpected raw issue types before mapping labels.
- Inspect class balance; a tiny example may not reflect real-world frequencies.
- Keep train, validation, and test splits separate when evaluating a model.
- Avoid leaking private customer details into training data; apply appropriate data minimization and anonymization.
- Validate entity boundaries and tags, especially for multi-token entities and sentences with multiple entities.
- Save the label-to-ID mapping and version it alongside the dataset and model.
- Evaluate outputs on held-out data before considering deployment.

## Key takeaway

Label engineering is the bridge between unstructured business records and consistent model-training examples. Classification labels, numeric IDs, prompt-target pairs, and token-level BIO tags serve different learning tasks; they should be designed and validated for the task they are intended to support.
