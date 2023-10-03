<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# Prodigy-ANN

![](images/approach.png)

This repository contains a Prodigy plugin for techniques that involve approximate nearest neighbors to fetch relevant subsets of the data to curate. To encode the text this library uses
[sentence-transformers](https://sbert.org) and it uses
[hnswlib](https://github.com/nmslib/hnswlib) as an index for these vectors.

You can install this plugin via `pip`. 

```
pip install "prodigy-ann @ git+https://github.com/explosion/prodigy-ann"
```

To learn more about this plugin, you can check the [Prodigy docs]().

