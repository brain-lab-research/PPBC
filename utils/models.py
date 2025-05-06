import torch.nn as nn
import torchvision.models as models


class ConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_mlp=None,
        output_dim=128,
        backbone_name="resnet18",
        starting_kernel_size=3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.hidden_mlp = hidden_mlp
        self.backbone_name = backbone_name

        self.backbone = getattr(models, self.backbone_name)(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=starting_kernel_size,
            stride=1,
            padding=1,
            bias=False,
        )

        self._remove_classifcation_layer()

        self._learning_type = "sl"  # ["sl", "ur"]

        self.projection_head = nn.Sequential(
            nn.Linear(
                in_features=self.in_features,
                out_features=self.hidden_mlp,
                bias=True,
            ),
            nn.BatchNorm1d(self.hidden_mlp),
            nn.SiLU(inplace=True),
            nn.Linear(
                in_features=self.hidden_mlp,
                out_features=self.output_dim,
                bias=True,
            ),
        )

        self.classification_head = nn.Linear(self.in_features, num_classes, bias=True)

    def _remove_classifcation_layer(self):
        if "resnet" in self.backbone_name:
            self.in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone {self.backbone} not supported yet.")

        self.hidden_mlp = self.hidden_mlp if self.hidden_mlp else self.in_features

    def supervised(self):
        self._learning_type = "sl"

    def unsupervised(self):
        self._learning_type = "ul"

    def forward(self, x):
        x = self.backbone(x)
        if self._learning_type == "sl":
            x = self.classification_head(x)
        elif self._learning_type == "ul":
            x = self.projection_head(x)
        else:
            raise ValueError(f"Invalid learning type: {self._learning_type}")
        return x
