import torch
import numpy as np


def get_loss(loss_cfg, df=None, device=None):
    loss_name = loss_cfg.loss_name
    loss = loss_cfg.config
    if loss_name == "ce":
        pos_weight = calculate_class_weights_multi_class(df)
        pos_weight = torch.tensor(pos_weight).to(device)
        return torch.nn.CrossEntropyLoss(
            weight=pos_weight,
            ignore_index=loss.ignore_index,
            reduction=loss.reduction,
            label_smoothing=loss.label_smoothing,
        )
    elif loss_name == "bce":
        if loss.pos_weight is None:
            loss.pos_weight = calculate_pos_weight(df)
        pos_weight = torch.tensor([loss.pos_weight]).to(device)
        return torch.nn.BCEWithLogitsLoss(
            reduction=loss.reduction,
            pos_weight=pos_weight,
        )
    else:
        raise ValueError("Unknown type of loss function")


def calculate_pos_weight(df):
    # Convert the 'target' column into a NumPy array
    target_array = np.array(df["target"].tolist())

    # Count zeros and ones for each index
    zeros_count = np.sum(target_array == 0, axis=0)
    ones_count = np.sum(target_array == 1, axis=0)

    # Calculate portion (ratio) of zeros to ones for each index
    ratios = zeros_count / ones_count
    return ratios.tolist()


def calculate_class_weights_multi_class(df):
    df_copy = df.copy()
    target_array = np.array(df_copy["target"].tolist())

    class_weights = {}
    ordered_weights = []

    unique_classes = np.unique(target_array)
    total_count = len(target_array)

    for cls in unique_classes:
        class_count = np.sum(target_array == cls, axis=0)
        class_weight = float(total_count / (len(unique_classes) * class_count))
        class_weights[cls] = class_weight

    # Ensure the weights are added to the list in order of ascending class index
    for cls in sorted(unique_classes):
        ordered_weights.append(class_weights[cls])

    return ordered_weights


class NTXentLoss(torch.nn.Module):
    """
    A NTXentLoss block that implements the Normalized Temperature-Scaled Cross Entropy Loss.

    :param device: The device on which computations will be performed.
    :param batch_size: The batch size.
    :param temperature: The temperature parameter for scaling.
    :param use_cosine_similarity: A boolean flag that determines whether cosine similarity is used. If False, dot product similarity is used.
    """

    def __init__(self, temperature, use_cosine_similarity, batch_size, device):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        """
        Returns the appropriate similarity function.

        :param use_cosine_similarity: A boolean flag that determines whether cosine similarity is used. If False, dot product similarity is used.

        :return: The similarity function.
        """
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        """
        Creates a mask for the samples from the same representation.

        :return: The mask tensor.
        """
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        """
        Computes the dot product similarity between two tensors.

        :param x: The first tensor.
        :param y: The second tensor.

        :return: The dot product similarity.
        """
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        """
        Computes the cosine similarity between two tensors.

        :param x: The first tensor.
        :param y: The second tensor.

        :return: The cosine similarity.
        """
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        """
        Computes the loss between two sets of representations.

        :param zis: The first set of representations.
        :param zjs: The second set of representations.

        :return: The computed loss.
        """
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
