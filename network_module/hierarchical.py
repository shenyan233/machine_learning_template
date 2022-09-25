import torch
from network_module.cluster_evaluate import cluster_elbow


class Hierarchical:
    def __init__(self, k=1):
        assert k > 0
        self.k = k
        self.leafs = None
        self.labels = None
        self.scores = None
        self.features = None

    def fit(self, x):
        self.features = x.clone()
        point_num = x.shape[0]
        self.leafs = torch.diag(torch.tensor([True] * point_num)).type_as(x).int()
        num_leaf = [1] * point_num
        active_id = [i for i in range(point_num)]
        self.labels = [1] * (point_num - self.k + 1)
        self.scores = -torch.zeros(point_num - self.k + 1).type_as(x)
        self.classify(active_id, -1)
        for index in range(point_num - self.k):
            current_features = x[active_id, :]
            num_active_point = len(active_id)
            distances = torch.norm(current_features[:, None] - current_features, dim=2, p=2)
            distances += torch.diag(torch.tensor([torch.inf] * num_active_point)).type_as(distances)
            index_dist = torch.argmin(distances)
            small_node_index = torch.div(index_dist, num_active_point, rounding_mode='trunc')
            big_node_index = index_dist % num_active_point
            new_vec = ((current_features[small_node_index, :] * num_leaf[
                active_id[small_node_index]] + current_features[big_node_index, :] * num_leaf[
                            active_id[big_node_index]]) / (
                               num_leaf[active_id[small_node_index]] + num_leaf[active_id[big_node_index]])).unsqueeze(
                dim=0)
            x = torch.cat((x, new_vec))
            self.leafs = torch.cat((self.leafs, (
                        self.leafs[active_id[small_node_index]] | self.leafs[active_id[big_node_index]]).unsqueeze(
                dim=0)), dim=0)
            num_leaf.append((num_leaf[active_id[small_node_index]] + num_leaf[active_id[big_node_index]]))
            # 一定要先del索引较大的
            del active_id[big_node_index]
            del active_id[small_node_index]
            active_id.append(x.shape[0] - 1)
            self.classify(active_id, -index - 2)
        return self.labels, self.scores

    def classify(self, active_id, nth_classfication):
        label = torch.argmax(self.leafs[active_id, :], dim=0)
        self.scores[nth_classfication] = cluster_elbow(self.features, label)
        self.labels[nth_classfication] = label


if __name__ == "__main__":
    a = torch.tensor([[0.95, 0.6], [1.03, 0.55], [1.2, 0.8], [4, 0.9], [4, 1.03], [2, 4.5], [2.1, 4.5]])
    clusters_list, scores = Hierarchical().fit(a)

    print('end')