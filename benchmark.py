import os
import numpy as np
import torch
from clustpy.data import load_optdigits, load_mnist, load_fmnist, load_usps, load_kmnist, load_cifar10, load_imagenet10, \
    load_imagenet_dog
from clustpy.deep import encode_batchwise, detect_device, get_dataloader, DEC, DCN, IDEC, DipEncoder, ACeDeC
from clustpy.deep.autoencoders import FeedforwardAutoencoder, ConvolutionalAutoencoder
from clustpy.deep._utils import embedded_kmeans_prediction
from clustpy.utils import EvaluationDataset, EvaluationAlgorithm, EvaluationMetric, EvaluationAutoencoder, \
    evaluate_multiple_datasets
from clustpy.metrics import unsupervised_clustering_accuracy as acc, \
    information_theoretic_external_cluster_validity_measure as dom
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, \
    adjusted_rand_score as ari
import inspect
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from clustpy.partition import SubKmeans
import torchvision

DOWNLOAD_PATH = None
SAVE_DIR = "ICDMBenchmark/"


class AEKmeans():

    def __init__(self, n_clusters, autoencoder=None, batch_size=256, random_state: np.random.RandomState = None,
                 custom_dataloaders: tuple = None):
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if self.custom_dataloaders is None:
            testloader = get_dataloader(X,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        drop_last=False)
        else:
            _, testloader = self.custom_dataloaders
        X_ae = encode_batchwise(testloader, self.autoencoder, torch.device('cpu'))
        km = KMeans(self.n_clusters)
        km.fit(X_ae)
        self.labels_ = km.labels_
        self.cluster_centers_ = km.cluster_centers_

    def predict(self, X: np.ndarray) -> np.ndarray:
        dataloader = get_dataloader(X, self.batch_size, False, False)
        ae = self.autoencoder.to(detect_device())
        predicted_labels = embedded_kmeans_prediction(dataloader, self.cluster_centers_, ae)
        return predicted_labels


class AESubKmeans():

    def __init__(self, n_clusters, autoencoder=None, batch_size=256, random_state: np.random.RandomState = None,
                 custom_dataloaders: tuple = None):
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if self.custom_dataloaders is None:
            testloader = get_dataloader(X,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        drop_last=False)
        else:
            _, testloader = self.custom_dataloaders
        X_ae = encode_batchwise(testloader, self.autoencoder, torch.device('cpu'))
        sk = SubKmeans(self.n_clusters)
        sk.fit(X_ae)
        self.labels_ = sk.labels_


class AESpectral():

    def __init__(self, n_clusters, autoencoder=None, batch_size=256, random_state: np.random.RandomState = None,
                 custom_dataloaders: tuple = None):
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if self.custom_dataloaders is None:
            testloader = get_dataloader(X,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        drop_last=False)
        else:
            _, testloader = self.custom_dataloaders
        X_ae = encode_batchwise(testloader, self.autoencoder, torch.device('cpu'))
        sc = SpectralClustering(self.n_clusters)
        sc.fit(X_ae)
        self.labels_ = sc.labels_


class AEEM():

    def __init__(self, n_clusters, autoencoder=None, batch_size=256, random_state: np.random.RandomState = None,
                 custom_dataloaders: tuple = None):
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if self.custom_dataloaders is None:
            testloader = get_dataloader(X,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        drop_last=False)
        else:
            _, testloader = self.custom_dataloaders
        X_ae = encode_batchwise(testloader, self.autoencoder, torch.device('cpu'))
        gmm = GaussianMixture(self.n_clusters, n_init=3)
        gmm.fit(X_ae)
        self.labels_ = gmm.predict(X_ae)


def _standardize(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    data = (data - mean) / std
    return data


def _add_color_channels_and_resize(data, conv_used, augmentation, dataset_name):
    if conv_used or augmentation:
        if data.ndim != 4:
            data = data.reshape(-1, 1, data.shape[1], data.shape[2])
            if conv_used:
                data = np.tile(data, (1, 3, 1, 1))
        if conv_used:
            if dataset_name in ["ImageNet10", "ImageNetDog"]:
                size = 224
            else:
                size = 32
            data = torchvision.transforms.Resize((size, size))(torch.from_numpy(data).float()).numpy()
    return data


def _get_dataset_loaders():
    datasets = [
        ("Optdigits", load_optdigits),
        ("USPS", load_usps),
        ("MNIST", load_mnist),
        ("FMNIST", load_fmnist),
        ("KMNIST", load_kmnist),
        ("CIFAR10", load_cifar10),
        ("ImageNet10", load_imagenet10),
        ("ImageNetDog", load_imagenet_dog)
    ]
    return datasets


def _get_dataloader_with_augmentation(data: np.ndarray, batch_size: int, flatten: int, data_name: str):
    data = torch.tensor(data)
    data /= 255.0
    channel_means = data.mean([0, 2, 3])
    channel_stds = data.std([0, 2, 3])
    # preprocessing functions
    normalize_fn = torchvision.transforms.Normalize(channel_means, channel_stds)
    flatten_fn = torchvision.transforms.Lambda(torch.flatten)
    # augmentation transforms
    if data_name in ["CIFAR10", "ImageNet10", "ImageNetDog"]:
        # color image augmentation according to SimCLR: https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/data_aug/contrastive_learning_dataset.py#L13
        _size = data.shape[-1]
        radias = int(0.1 * _size) // 2
        kernel_size = radias * 2 + 1
        color_jitter = torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        transform_list = [
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.RandomResizedCrop(size=_size),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomApply([color_jitter], p=0.8),
                        torchvision.transforms.RandomGrayscale(p=0.2),
                        torchvision.transforms.GaussianBlur(kernel_size=kernel_size),
                        torchvision.transforms.ToTensor(),
                        normalize_fn,
        ]
    else:
        transform_list = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomAffine(degrees=(-16, +16), translate=(0.1, 0.1), shear=(-8, 8), fill=0),
            torchvision.transforms.ToTensor(),
            normalize_fn
        ]
    orig_transform_list = [normalize_fn]
    if flatten:
        transform_list.append(flatten_fn)
        orig_transform_list.append(flatten_fn)
    aug_transforms = torchvision.transforms.Compose(transform_list)
    orig_transforms = torchvision.transforms.Compose(orig_transform_list)
    # pass transforms to dataloader
    aug_dl = get_dataloader(data, batch_size=batch_size, shuffle=True,
                            ds_kwargs={"aug_transforms_list": [aug_transforms],
                                       "orig_transforms_list": [orig_transforms]})
    orig_dl = get_dataloader(data, batch_size=batch_size, shuffle=False,
                             ds_kwargs={"orig_transforms_list": [orig_transforms]})
    return aug_dl, orig_dl


def _get_evaluation_datasets_with_autoencoders(dataset_loaders, ae_layers, experiment_name, n_repetitions, batch_size,
                                               n_pretrain_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                                               ae_class, other_ae_params, device, augmentation=False,
                                               train_test_split=False):
    evaluation_datasets = []
    # Get autoencoders for DC algortihms
    for data_name_orig, data_loader in dataset_loaders:
        data_name_exp = data_name_orig + "_" + experiment_name
        data_loader_params = inspect.getfullargspec(data_loader).args
        flatten = (ae_class != ConvolutionalAutoencoder and not augmentation)
        train_subset = {"subset": "train"} if train_test_split else {}
        if augmentation:
            assert flatten == False, "If augmentation is used, flatten must be false"
            # Normalization happens within the augmentation dataloader
            data, labels = data_loader(flatten=flatten, downloads_path=DOWNLOAD_PATH, **train_subset)
        elif "normalize_channels" in data_loader_params and not train_test_split:
            data, labels = data_loader(flatten=flatten, normalize_channels=True, downloads_path=DOWNLOAD_PATH,
                                       **train_subset)
        else:
            data, labels = data_loader(flatten=flatten, downloads_path=DOWNLOAD_PATH, **train_subset)
            data_mean = np.mean(data)
            data_std = np.std(data)
            data = _standardize(data, data_mean, data_std)
        # Change data format if conv autoencoder is used
        conv_used = ae_class == ConvolutionalAutoencoder
        data = _add_color_channels_and_resize(data, conv_used, augmentation, data_name_orig)
        # Create dataloaders
        if augmentation:
            flatten = ae_class != ConvolutionalAutoencoder  # Update flatten -> should still happen if AE is not Conv
            if not os.path.isdir(SAVE_DIR + "{0}/DLs".format(data_name_exp)):
                os.makedirs(SAVE_DIR + "{0}/DLs".format(data_name_exp))
            save_path_dl1_aug = SAVE_DIR + "{0}/DLs/dl_aug.pth".format(data_name_exp)
            dataloader, orig_dataloader = _get_dataloader_with_augmentation(data, batch_size, flatten, data_name_orig)
            if not os.path.isfile(save_path_dl1_aug):
                torch.save(dataloader, save_path_dl1_aug)
            save_path_dl1_orig = SAVE_DIR + "{0}/DLs/dl_orig.pth".format(data_name_exp)
            if not os.path.isfile(save_path_dl1_orig):
                torch.save(orig_dataloader, save_path_dl1_orig)
            path_custom_dataloaders = (save_path_dl1_aug, save_path_dl1_orig)
        else:
            dataloader = get_dataloader(data, batch_size, shuffle=True)
            path_custom_dataloaders = None
        evaluation_autoencoders = []
        for i in range(n_repetitions):
            # Pretrain and save autoencoder
            save_path_ae = SAVE_DIR + "{0}/AEs/ae_{1}.ae".format(data_name_exp, i)
            if ae_class != ConvolutionalAutoencoder:
                layers = [data[0].size] + ae_layers
                ae_params = dict(**other_ae_params, **{"layers": layers})
            else:
                ae_params = dict(**other_ae_params, **{"fc_layers": ae_layers, "input_height": data.shape[-1]})
            if not os.path.isfile(save_path_ae):
                ae = ae_class(**ae_params).to(device)
                ae.fit(n_pretrain_epochs, pretrain_optimizer_params, batch_size, dataloader=dataloader,
                       optimizer_class=optimizer_class,
                       loss_fn=loss_fn, device=device, model_path=save_path_ae)
                print("created autoencoder {0} for dataset {1}".format(i, data_name_exp))
            eval_autoencoder = EvaluationAutoencoder(save_path_ae, ae_class, ae_params, path_custom_dataloaders)
            evaluation_autoencoders.append(eval_autoencoder)
        if augmentation:
            eval_dataset = EvaluationDataset(data_name_exp, data_loader,
                                             data_loader_params={"flatten": False,
                                                                 "downloads_path": DOWNLOAD_PATH},
                                             iteration_specific_autoencoders=evaluation_autoencoders,
                                             train_test_split=train_test_split,
                                             preprocess_methods=_add_color_channels_and_resize,
                                             preprocess_params={"conv_used": conv_used, "augmentation": augmentation,
                                                                "dataset_name": data_name_orig})
        elif "normalize_channels" in data_loader_params and not train_test_split:
            eval_dataset = EvaluationDataset(data_name_exp, data_loader,
                                             data_loader_params={"normalize_channels": True, "flatten": flatten,
                                                                 "downloads_path": DOWNLOAD_PATH},
                                             iteration_specific_autoencoders=evaluation_autoencoders,
                                             train_test_split=train_test_split,
                                             preprocess_methods=_add_color_channels_and_resize,
                                             preprocess_params={"conv_used": conv_used, "augmentation": augmentation,
                                                                "dataset_name": data_name_orig})
        else:
            eval_dataset = EvaluationDataset(data_name_exp, data_loader,
                                             data_loader_params={"flatten": flatten, "downloads_path": DOWNLOAD_PATH},
                                             iteration_specific_autoencoders=evaluation_autoencoders,
                                             train_test_split=train_test_split,
                                             preprocess_methods=[_standardize, _add_color_channels_and_resize],
                                             preprocess_params=[{"mean": data_mean, "std": data_std},
                                                                {"conv_used": conv_used, "augmentation": augmentation,
                                                                 "dataset_name": data_name_orig}])
        evaluation_datasets.append(eval_dataset)
    return evaluation_datasets


def _get_evaluation_algorithms(n_clustering_epochs, embedding_size, batch_size, optimizer_class, loss_fn,
                               augmentation=False):
    evaluation_algorithms = [
        EvaluationAlgorithm("AE+KMeans", AEKmeans,
                            {"n_clusters": None, "batch_size": batch_size}),
        EvaluationAlgorithm("DEC", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),
        EvaluationAlgorithm("IDEC", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),
        EvaluationAlgorithm("DCN", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),
        EvaluationAlgorithm("ACeDeC", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),
        EvaluationAlgorithm("DipEncoder", DipEncoder,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation})
    ]
    return evaluation_algorithms


def _get_evaluation_metrics():
    evaluation_metrics = [
        EvaluationMetric("NMI", nmi),
        EvaluationMetric("AMI", ami),
        EvaluationMetric("ARI", ari),
        EvaluationMetric("ACC", acc),
        EvaluationMetric("DOM", dom),
    ]
    return evaluation_metrics


def _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, augmentation=False, train_test_split=False):
    ae_layers = ae_layers.copy()
    ae_layers.append(embedding_size)
    experiment_name = experiment_name + "_" + "_".join(str(x) for x in ae_layers)
    dataset_loaders = _get_dataset_loaders()
    device = detect_device()
    evaluation_datasets = _get_evaluation_datasets_with_autoencoders(dataset_loaders, ae_layers, experiment_name,
                                                                     n_repetitions, batch_size, n_pretrain_epochs,
                                                                     optimizer_class, pretrain_optimizer_params,
                                                                     loss_fn, ae_class, other_ae_params,
                                                                     device, augmentation, train_test_split)
    evaluation_algorithms = _get_evaluation_algorithms(n_clustering_epochs, embedding_size, batch_size, optimizer_class,
                                                       loss_fn, augmentation)
    evaluation_metrics = _get_evaluation_metrics()
    evaluate_multiple_datasets(evaluation_datasets, evaluation_algorithms, evaluation_metrics, n_repetitions,
                               add_runtime=True, add_n_clusters=False,
                               save_path=SAVE_DIR + experiment_name + "/Results/result.csv",
                               save_intermediate_results=True,
                               save_labels_path=SAVE_DIR + experiment_name + "/Labels/label.csv")


def experiment_feedforward_512_256_128_10(n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                          n_clustering_epochs=150,
                                          optimizer_class=torch.optim.Adam,
                                          pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                          other_ae_params={}):
    experiment_name = "FF"
    embedding_size = 10
    ae_layers = [512, 256, 128]
    ae_class = FeedforwardAutoencoder
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params)


def experiment_feedforward_500_500_2000_10(n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                           n_clustering_epochs=150,
                                           optimizer_class=torch.optim.Adam,
                                           pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                           other_ae_params={}):
    experiment_name = "FF"
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params)


def experiment_feedforward_500_500_2000_10_aug(n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                               n_clustering_epochs=150,
                                               optimizer_class=torch.optim.Adam,
                                               pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                               other_ae_params={}):
    experiment_name = "FF_AUG"
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder
    augmentation = True
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, augmentation)


def experiment_embedding_size(n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                              n_clustering_epochs=150,
                              optimizer_class=torch.optim.Adam,
                              pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                              other_ae_params={}):
    experiment_name = "EMBEDDING_SIZE"
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder
    for embedding_size in [2, 5, 25, 50, 100]:
        _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                    n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                    ae_class, other_ae_params)


def experiment_convolutional_resnet18(n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                      n_clustering_epochs=150,
                                      optimizer_class=torch.optim.Adam,
                                      pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                      other_ae_params={"conv_encoder_name": "resnet18"}):
    experiment_name = "CONV_RESNET18"
    embedding_size = 10
    ae_layers = [512]
    ae_class = ConvolutionalAutoencoder
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params)


def experiment_convolutional_resnet18_aug(n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                          n_clustering_epochs=150,
                                          optimizer_class=torch.optim.Adam,
                                          pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                          other_ae_params={"conv_encoder_name": "resnet18"}):
    experiment_name = "CONV_RESNET18_AUG"
    embedding_size = 10
    ae_layers = [512]
    ae_class = ConvolutionalAutoencoder
    augmentation = True
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, augmentation)


def experiment_learning_rate(n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                             n_clustering_epochs=150,
                             optimizer_class=torch.optim.Adam,
                             pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                             other_ae_params={}):
    experiment_name = "FF"
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder

    ae_layers.append(embedding_size)
    experiment_name = experiment_name + "_" + "_".join(str(x) for x in ae_layers)
    dataset_loaders = [
        ("USPS", load_usps),
        ("MNIST", load_mnist),
        ("ImageNet10", load_imagenet10)
    ]
    device = detect_device()
    evaluation_datasets = _get_evaluation_datasets_with_autoencoders(dataset_loaders, ae_layers, experiment_name,
                                                                     n_repetitions, batch_size, n_pretrain_epochs,
                                                                     optimizer_class, pretrain_optimizer_params,
                                                                     loss_fn, ae_class, other_ae_params,
                                                                     device)
    evaluation_algorithms = [
        EvaluationAlgorithm("DEC_LR_1e-5", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DEC_LR_5e-5", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DEC_LR_5e-4", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DEC_LR_1e-3", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-3}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("IDEC_LR_1e-5", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("IDEC_LR_5e-5", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("IDEC_LR_5e-4", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("IDEC_LR_1e-3", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-3}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DCN_LR_1e-5", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DCN_LR_5e-5", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DCN_LR_5e-4", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DCN_LR_1e-3", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-3}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("ACeDeC_LR_1e-5", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("ACeDeC_LR_5e-5", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("ACeDeC_LR_1e-4", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("ACeDeC_LR_1e-3", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-3}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DipEncoder_LR_1e-5", DipEncoder,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DipEncoder_LR_5e-5", DipEncoder,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DipEncoder_LR_5e-4", DipEncoder,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DipEncoder_LR_1e-3", DipEncoder,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-3}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
    ]
    evaluation_metrics = _get_evaluation_metrics()
    evaluate_multiple_datasets(evaluation_datasets, evaluation_algorithms, evaluation_metrics, n_repetitions,
                               add_runtime=True, add_n_clusters=False,
                               save_path=SAVE_DIR + experiment_name + "/Results/result_lr.csv",
                               save_intermediate_results=True,
                               save_labels_path=SAVE_DIR + experiment_name + "/Labels/label_lr.csv")


def experiment_initial_clustering(n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                  n_clustering_epochs=150,
                                  optimizer_class=torch.optim.Adam,
                                  pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                  other_ae_params={}):
    experiment_name = "FF"
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder

    ae_layers.append(embedding_size)
    experiment_name = experiment_name + "_" + "_".join(str(x) for x in ae_layers)
    dataset_loaders = [
        ("USPS", load_usps),
        ("MNIST", load_mnist),
        ("ImageNet10", load_imagenet10)
    ]
    device = detect_device()
    evaluation_datasets = _get_evaluation_datasets_with_autoencoders(dataset_loaders, ae_layers, experiment_name,
                                                                     n_repetitions, batch_size, n_pretrain_epochs,
                                                                     optimizer_class, pretrain_optimizer_params,
                                                                     loss_fn, ae_class, other_ae_params,
                                                                     device)
    evaluation_algorithms = [
        EvaluationAlgorithm("AE+EM", AEEM,
                            {"n_clusters": None, "batch_size": batch_size}),
        EvaluationAlgorithm("DEC+EM", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": GaussianMixture,
                             "initial_clustering_params": {"n_init": 3}}),
        EvaluationAlgorithm("IDEC+EM", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": GaussianMixture,
                             "initial_clustering_params": {"n_init": 3}}),
        EvaluationAlgorithm("DCN+EM", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": GaussianMixture,
                             "initial_clustering_params": {"n_init": 3}}),
        EvaluationAlgorithm("AE+Spectral", AESpectral,
                            {"n_clusters": None, "batch_size": batch_size}),
        EvaluationAlgorithm("DEC+Spectral", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SpectralClustering}),
        EvaluationAlgorithm("IDEC+Spectral", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SpectralClustering}),
        EvaluationAlgorithm("DCN+Spectral", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SpectralClustering}),
        EvaluationAlgorithm("AE+SubKmeans", AESubKmeans,
                            {"n_clusters": None, "batch_size": batch_size}),
        EvaluationAlgorithm("DEC+SubKmeans", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SubKmeans}),
        EvaluationAlgorithm("IDEC+SubKmeans", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SubKmeans}),
        EvaluationAlgorithm("DCN+SubKmeans", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SubKmeans})
    ]
    evaluation_metrics = _get_evaluation_metrics()
    evaluate_multiple_datasets(evaluation_datasets, evaluation_algorithms, evaluation_metrics, n_repetitions,
                               add_runtime=True, add_n_clusters=False,
                               save_path=SAVE_DIR + experiment_name + "/Results/result_init_clust.csv",
                               save_intermediate_results=True,
                               save_labels_path=SAVE_DIR + experiment_name + "/Labels/label_init_clust.csv")


def experiment_feedforward_test_train(n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                      n_clustering_epochs=150,
                                      optimizer_class=torch.optim.Adam,
                                      pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                      other_ae_params={}):
    experiment_name = "TEST_TRAIN"
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, train_test_split=True)


if __name__ == "__main__":
    # experiment_feedforward_512_256_128_10()
    # experiment_feedforward_500_500_2000_10()
    # experiment_feedforward_500_500_2000_10_aug()
    # experiment_embedding_size()
    # experiment_convolutional_resnet18()
    # experiment_convolutional_resnet18_aug()
    # experiment_initial_clustering()
    # experiment_learning_rate()
    experiment_feedforward_test_train()
