import numpy as np

def read_ivecs(filename):
    """
    Reads an ivecs file and returns a numpy array of shape (num_vectors, dim)
    """
    with open(filename, 'rb') as f:
        raw = np.fromfile(f, dtype=np.int32)

    dims = raw[0]  # all vectors have same dimension
    vectors = []
    i = 0
    while i < len(raw):
        d = raw[i]
        assert d == dims, f"Dimension mismatch at vector {len(vectors)}"
        vector = raw[i+1:i+1+d]
        vectors.append(vector)
        i += d + 1
    return np.array(vectors, dtype=np.int32)

def compute_recall(groundtruth, results, k=10):
    """
    Compute recall@k for a set of queries
    groundtruth: np.array of shape (num_queries, >=k)
    results: np.array of shape (num_queries, >=k)
    """
    num_queries = groundtruth.shape[0]
    matches = 0
    for i in range(num_queries):
        gt_set = set(groundtruth[i, :k])
        res_set = set(results[i, :k])
        matches += len(gt_set & res_set)
    print(matches)
    recall = matches / (num_queries * k)
    return recall

if __name__ == "__main__":
    gt_file = "./benchmarks/siftsmall/siftsmall_groundtruth.ivecs"
    res_file = "./knn_out.ivecs"

    groundtruth = read_ivecs(gt_file)
    results = read_ivecs(res_file)

    recall = compute_recall(groundtruth, results, k=10)
    print(f"Recall@10: {recall*100:.2f}%")
