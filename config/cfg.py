cfg = {
    'gpu_id': 0,
    'max_len': 100,
    'train_batch_size': 1,
    'test_batch_size': 32,
    'bert_learning_rate': 1e-5,
    'other_learning_rate': 1e-5,
    'epoch': 20,
    'K': 8,
    'Kt': 2000,
    'template': 'It was <mask>.',
    'answer': ['terrible', 'great'],
    'device': 'cuda',
    'optimizer': 'Adam',
    'word_size': 50265
}

hyper_roberta = {
    'word_dim': 1024,
    'dropout': 0.1
}

path = {
    'neg_path': 'data/rt-polaritydata/neg_label.txt',
    'pos_path': 'data/rt-polaritydata/pos_label.txt',
    'roberta_path': 'pretrained_models/roberta-large'
}


