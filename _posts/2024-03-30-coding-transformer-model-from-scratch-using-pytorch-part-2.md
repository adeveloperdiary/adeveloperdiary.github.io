---
title: Coding Transformer Model from Scratch Using PyTorch - Part 2 (Data Processing and Preparation)
categories: [data-science, deep-learning, nlp]
tags: [NLP]
math: true
---

Welcome back to the second installment of our series on coding a Transformer model from scratch using PyTorch! In this part, we'll dive into the crucial aspect of data processing and preparation. Handling data efficiently is paramount for any machine learning task, and building a Transformer model is no exception. We'll guide you through the step-by-step process of downloading the data and performing essential preprocessing tasks such as tokenization and padding using PyTorch. By the end of this tutorial, you'll have a solid understanding of how to preprocess your data effectively, setting the stage for training your Transformer model. So, let's roll up our sleeves and get started on this data preprocessing journey!

## Tokenizer

To begin, let's install two essential packages:

```shell
pip install datasets
pip install transformers
```

For our tokenizer, I suggest starting with the `opus_books` [https://huggingface.co/datasets/opus_books] dataset from the `huggingface` library. However, feel free to experiment with larger datasets if you aim to build a more robust model. For context, I'm training the model using a NVIDIA 2080ti GPU with 11GB of available RAM.

### Downloading Data

You can easily download the dataset directly from `huggingface` using the `load_dataset()` function from the `datasets` package. The first argument specifies the dataset's name, while the second argument defines the two different languages for translation. In this case, I'm downloading the English and French language translations. The last argument determines the available set we require, such as train, test, or validation. Since our focus is on learning, we'll simply download the `train` file and subsequently split it into `train` and `validation` sets as necessary.

```python
from datasets import load_dataset
hf_dataset=load_dataset('opus_books',f'en-fr',split='train')
```

If you just print the `hf_dataset`, it will show total number of rows and the feature names.

```
Dataset({
    features: ['id', 'translation'],
    num_rows: 127085
})
```

We can access this as `dict` by passing the feature name and index of any element.

```python
hf_dataset['translation'][0]
```

```
{'en': 'The Wanderer', 'fr': 'Le grand Meaulnes'}
```

### Dataset Iterator

As observed, to efficiently process the entire dataset, we require a function to iterate through each translation in a lazy manner. We can accomplish this by utilizing `yield` instead of `return` to create a generator function. Below is the `get_one_sentence()` function, which takes in the entire dataset and the language to retrieve the sentence from. By employing `yield`, the function produces a value while retaining its state, enabling it to resume from where it left off. This approach eliminates the need to duplicate the dataset in memory, enhancing efficiency.

```python
def traverse_sentences(dataset, lang):
    for row in dataset:
        yield row['translation'][lang]
```

We can test the above function to make sure its working as expected.

```python
print(next(traverse_sentences(hf_dataset,'en')))
print(next(traverse_sentences(hf_dataset,'fr')))
```

```
The Wanderer
Le grand Meaulnes
```

### Tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
```

Create an instance of the `WordLevel` `Tokenizer` and set `[UNK]` for unknown tokens. 

```python
tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
```

We want to split the sentences by white space before applying the tokenizer. Hence set the `pre_tokenizer` to `Whitespace()`

```python
tokenizer.pre_tokenizer=Whitespace()
```

Now we need to define the trainer. We will use 4 special tokens `"[UNK]","[PAD]","[SOS]","[EOS]"` for Unknown tokens, Paddings, Start & End of Sentences. Also set `min_frequency` to `2 ` or `3` to accept words when their frequency of occurrence is at least equal to `min_frequency`.

```python
trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
```

In order to train the `tokenizer` , we need to pass all the sentences one by one. We can use the `traverse_sentences` function. 

Invoke `train_from_iterator()` function by passing the function `get_one_sentence` and the `trainer`. Then `save` the tokenizer to local path. This code is very mostly copied from the documentation of `tokenizer` library. You can always refer the official documentation for additional details.

```python
tokenizer.train_from_iterator(traverse_sentences(dataset,lang),trainer=trainer)
tokenizer.save(str(tokenizer_path))
```

##### Let's put everything together in a single function. We will load the tokenizer if its already created otherwise we will create one. 

```python
from pathlib import Path
def init_tokenizer(dataset, lang):
    tokenizer_path = Path(f"tokenizer_{lang}.json")
    tokenizer_loaded = False

    if tokenizer_path.exists():
        # try loading tokenizer if exists
        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer_loaded = True
        except:
            pass
    if not tokenizer_loaded:
        # initiate tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            traverse_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    return tokenizer
```

Question is why we need the tokenizer instance. The `Tokenizer` includes many built-in functions which can be directly invoked. Here is a glimpse. The most important ones will be `token_to_id()` and `id_to_token()`.

<img src="../assets/img/coding-transformer-model-from-scratch-using-pytorch-part-2-adeveloperdiary.jpg" alt="coding-transformer-model-from-scratch-using-pytorch-part-2-adeveloperdiary.jpg" style="zoom:50%;" />

Lets, see it in practice. The example code is self-explanatory. 

```python
# Get the vocabulary size
print(tokenizer_src.get_vocab_size())
# Get the word for random id
print(tokenizer_src.id_to_token(1234))
# Get the id for "learning"
print(tokenizer_src.token_to_id("learning"))
# Convert sentence to array of token ids
print(tokenizer_src.encode("i love learning").ids)
# Convert array of token ids to sentence.
print(tokenizer_src.decode([5552, 194, 3125]))
```

```
15698
pay
3125
[5552, 194, 3125]
i love learning
```





