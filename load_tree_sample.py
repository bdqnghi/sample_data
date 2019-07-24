import os
import util
import gzip
import pickle


data_directory = "ProgramData_pb_slice_pkl"
pretrained_embeddings_url = "embedding/fast_pretrained_vectors.pkl.gz"

all_files = []
all_labels = []
for root,directories,files in os.walk(data_directory):
    for file in files:
        try:
            file_path = os.path.join(root,file)
            all_files.append(file_path)

            splits = file_path.split("/")
            # l = splits[len(splits)-2]
            # print(splits)
           
            label = splits[len(splits)-2]
            all_labels.append(int(label))
            #     ast_representation = build_tree(file_path)
            #     result.append({
            #         'tree': ast_representation, 'metadata': {'label': label}
            #     })
        except Exception as err:
            print(err)

files_with_labels = zip(all_files, all_labels)

all_labels_unique = set(all_labels)
print(all_labels)

print("Loading pretrained embeddings..........")
with gzip.open(pretrained_embeddings_url, 'rb') as fh:
    embeddings, embed_lookup = pickle.load(fh,encoding='latin1')
  
util.gen_samples(files_with_labels, all_labels_unique, embeddings, embed_lookup)