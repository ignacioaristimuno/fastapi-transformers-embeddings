# Endpoints Documentation

## 1. Example of Endpoint for Text Classification

### Description

This endpoint allows users to encode a list of texts and receive the embeddings associated to those texts.

### Endpoint

```
POST /encoding/texts
```

### Request Body

```json
{
  "texts": [
    "This is an example of a sentence",
    "Another example of a another sentence"
  ],
  "instruct": false
}
```
**Note:** the `instruct` argument is established as some of the newest embedding models allow instructions to prefix the text mainly for retrieval purposes. By default this is set to `False`. In case you are using an instruction-based embedding model, fill your VectorDB with `instruct=False`, but set `instruct=True` for retrieval.

### Response Body

```json
{
  "texts": [
    {
      "text": "This is an example of a sentence",
      "embedding": [
        0.02410719357430935,
        ...
        0.013726074025034904
      ]
    },
    {
      "text": "Another example of a another sentence",
      "embedding": [
        0.017397478222846985,
        ...
        -0.0009440501453354955
      ]
    }
  ],
  "model_id": "intfloat/multilingual-e5-large-instruct",
  "embedding_dim": 1024
}

```

### Notes

- The `texts` field in the request body should contain a list of strings representing the texts to be encoded.
- The response contains a list of dictionaries, each representing a text along with its embedding.
- The `model_id` field in the response indicates the identifier of the model used for generating the embeddings.
- The `embedding_dim` field in the response indicates the size of the generated embeddings.
