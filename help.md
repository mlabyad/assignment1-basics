mkdir -p data
cd data

curl -L -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
curl -L -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

curl -L -o owt_train.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
curl -L -o owt_valid.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

gunzip owt_valid.txt.gz
gunzip owt_train.txt.gz

cd ..