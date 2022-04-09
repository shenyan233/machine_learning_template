import torch


class Kmeans:
    def __init__(self, n_clusters, max_iter=None, verbose=False, device=torch.device("cpu")):
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        unique_centers = torch.unique(init_row, dim=0)
        init_points = x[unique_centers]
        res_points = x.clone()
        while init_points.shape[0] != self.n_clusters:
            points_index = torch.zeros(res_points.shape[0]).to(self.device)
            points_index[unique_centers] = 1
            res_points = res_points[points_index == 0]
            num_need_centers = self.n_clusters - init_points.shape[0]
            init_row = torch.randint(0, res_points.shape[0], (num_need_centers,)).to(self.device)
            unique_centers = torch.unique(init_row, dim=0)
            init_points = torch.cat((init_points, res_points[unique_centers]), dim=0)
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, 0))
            if torch.abs(self.variation) < 1e-4 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        return self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), 1)
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], 0)
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)  # shape (0, 250000)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, 0).unsqueeze(0)], 0)

        self.centers = centers  # shape (10, 2048)

    def representative_sample(self):
        self.representative_samples = torch.argmin(self.dists, 1)
        return self.representative_samples
