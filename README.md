# BERT for Aspect-Based Sentiment Analysis


## Running


Execute the following command to run the model for Aspect Extractio (NER) task:

```!python src/run_ae.py --data_dir "ae/laptop" --eval_data_dir "ae/rest" --output_dir "pt_model/laptop_ae" --do_train --num_train_epochs 10 --do_valid```


Execute the following command to run the model for Aspect Sentiment Classification task:

```!python src/run_asc.py --data_dir "asc/laptop" --eval_data_dir "asc/rest" --output_dir "pt_model/laptop_asc" --do_train --num_train_epochs 10 --do_eval --do_valid```

![Poster](https://github.com/pulkit6559/bert-absc/blob/main/Project%20Poster%20-%20Team%2030.png?raw=true)


