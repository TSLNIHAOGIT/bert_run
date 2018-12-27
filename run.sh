export BERT_BASE_DIR=/Users/ozintel/Downloads/Tsl_python_progect/local_ml/bert/pretrain_model/chinese_L-12_H-768_A-12 #全局变量 下载的预训练bert地址
export MY_DATASET=/Users/ozintel/Downloads/Tsl_python_progect/local_ml/bert/data_examples #全局变量 数据集所在地址
python3.6 run_classifier.py \
  --task_name=senti_analysis \
  #--do_train=True \
  #--do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  #--train_batch_size=32 \
  #--learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=/Users/ozintel/Downloads/Tsl_python_progect/local_ml/bert/outs