import torch


def cluster_silhouette(features, clusters_result):
    silhouette = torch.zeros(1).type_as(features)
    for index_cluster, feature in zip(clusters_result, features):
        cluster = features[clusters_result == index_cluster]
        a = torch.zeros(1).type_as(features)
        for other_feature in cluster:
            if cluster.shape[0] != 1:
                a = torch.sum(torch.sum((other_feature - cluster) ** 2, dim=1) ** 0.5) / (cluster.shape[0] - 1)
        b = torch.inf
        for index_cluster_other in range(torch.max(clusters_result).int() + 1):
            if index_cluster != index_cluster_other:
                other_cluster = features[clusters_result == index_cluster_other]
                b_temp = torch.mean(torch.sum((feature - other_cluster) ** 2, dim=1) ** 0.5)
                if b_temp < b:
                    b = b_temp
        s = (b - a) / torch.max(a, b)
        silhouette += s
    return silhouette / features.shape[0]


def cluster_ch(features, clusters_result):
    n_samples = features.shape[0]
    n_labels = torch.max(clusters_result) + 1

    extra_disp, intra_disp = 0., 0.
    mean = torch.mean(features, dim=0)
    for k in range(n_labels):
        cluster_k = features[clusters_result == k]
        mean_k = torch.mean(cluster_k, dim=0)
        extra_disp += cluster_k.shape[0] * torch.sum((mean_k - mean) ** 2)
        intra_disp += torch.sum((cluster_k - mean_k) ** 2)
    return torch.ones(1).type_as(intra_disp) if intra_disp == 0. else \
        (extra_disp * (n_samples - n_labels) /
         (intra_disp * (n_labels - 1.))).unsqueeze(dim=0)


def cluster_elbow(features, clusters_result):
    n_labels = torch.max(clusters_result) + 1
    intra_disp = torch.zeros(1)
    for k in range(n_labels):
        cluster_k = features[clusters_result == k]
        mean_k = torch.mean(cluster_k, dim=0)
        intra_disp += torch.sum((cluster_k - mean_k) ** 2)
    return intra_disp
