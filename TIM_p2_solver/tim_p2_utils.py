import torch
from torch import Tensor, nn
from TIM_p2_solver.few_shot_classifier import FewShotClassifier
from TIM_p2_solver.utils import get_one_hot

class TIM_p2(FewShotClassifier):
    """
    Fine-tune prototypes based on
        1) classification error on support images
        2) mutual information between query features and their label predictions
        3) Kullback-Leibler(KL) divergence-based loss between the model prediction
           and the zero-shot CLIP outputs. 
    Classify w.r.t. to euclidean distance to updated prototypes.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    TIM_p2 is a transductive method.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 150,
        fine_tuning_lr: float = 1e-4,
        cross_entropy_weight: float = 0.1,
        marginal_entropy_weight: float = 1.0,
        conditional_entropy_weight: float = 0.1,
        temperature: float = 15.0,
        alpha: float = 1.0,
        gamma: float = 1,
        **kwargs,
    ):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            cross_entropy_weight: weight given to the cross-entropy term of the loss
            marginal_entropy_weight: weight given to the marginal entropy term of the loss
            conditional_entropy_weight: weight given to the conditional entropy term of the loss
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
            gamma: weight given to the KL divergence term with textual information of the loss
        """
        super().__init__(*args, **kwargs)

        # Since we fine-tune the prototypes we need to make them leaf variables
        # i.e. we need to freeze the backbone.
        self.backbone.requires_grad_(False)

        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.cross_entropy_weight = cross_entropy_weight
        self.marginal_entropy_weight = marginal_entropy_weight
        self.conditional_entropy_weight = conditional_entropy_weight
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma

        self.loss_weights = [self.cross_entropy_weight, self.marginal_entropy_weight, self.conditional_entropy_weight]


    def q_update(self, P, clip_logits):
        """
        inputs:
            P : torch.tensor of shape [n_tasks, q_shot, num_class]
                where P[i,j,k] = probability of point j in task i belonging to class k
                (according to our L2 classifier)
            clip_logits : torch.Tensor of shape [n_task, n, num_class]
        """
        l1, l2 = self.loss_weights[1], self.loss_weights[2]
        l3 = 1.0  # Corresponds to the weight of the KL penalty
        alpha = l2 / l3
        beta = l1 / (l1 + l3)

        Q = ((P ** (1+alpha)) * (clip_logits ** self.gamma)) / (((P ** (1+alpha)) * (clip_logits ** self.gamma)).sum(dim=1, keepdim=True)) ** beta
        self.Q = (Q / Q.sum(dim=2, keepdim=True)).float()

    def weights_update(self, support, query, y_s_one_hot):
        """
        Corresponds to w_k updates
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, s_shot, num_classes]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        P_s = self.get_logits(support).softmax(2)
        P_q = self.get_logits(query).softmax(2)
        src_part = self.loss_weights[0] / (1 + self.loss_weights[2]) * y_s_one_hot.transpose(1, 2).matmul(support)
        src_part += self.loss_weights[0] / (1 + self.loss_weights[2]) * (self.weights * P_s.sum(1, keepdim=True).transpose(1, 2)\
                                                                         - P_s.transpose(1, 2).matmul(support))
        src_norm = self.loss_weights[0] / (1 + self.loss_weights[2]) * y_s_one_hot.sum(1).view(n_tasks, -1, 1)

        qry_part = self.N_s / self.N_q * self.Q.transpose(1, 2).matmul(query)
        qry_part += self.N_s / self.N_q * (self.weights * P_q.sum(1, keepdim=True).transpose(1, 2)\
                                           - P_q.transpose(1, 2).matmul(query))
        qry_norm = self.N_s / self.N_q * self.Q.sum(1).view(n_tasks, -1, 1)

        new_weights = (src_part + qry_part) / (src_norm + qry_norm)
        self.weights = self.weights + self.alpha * (new_weights - self.weights)

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = self.temperature * (samples.matmul(self.weights.transpose(1, 2)) \
                              - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds


    def init_weights(self, support, y_s, query, clip_logits):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot, num_classes]
            clip_logits : torch.Tensor of shape [n_task, q_shot, num_classes]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        one_hot = y_s
        
        y_text = clip_logits
        max_indices = y_text.argmax(dim=-1)
        one_hot_text = torch.zeros_like(y_text)
        tasks, n_query = max_indices.shape
        one_hot_text[torch.arange(tasks)[:, None], torch.arange(n_query)[None, :], max_indices] = 1

        counts = one_hot.sum(1).view(n_tasks, -1, 1).type(torch.FloatTensor).to(self.support_features.device)
        counts_text = one_hot_text.sum(1).view(tasks, -1, 1).type(torch.FloatTensor).to(self.support_features.device)
        weights = one_hot.transpose(1, 2).matmul(support)
        weights_text = one_hot_text.transpose(1, 2).matmul(query)
        self.weights = (weights + weights_text) / (counts + counts_text)


    def forward(
        self,
        query_images: Tensor,
        clip_weights: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune prototypes based on support classification error, mutual information between
        query features and their label predictions, and a zero-shot CLIP-based KL-divergence
        regularizer.
        Then classify w.r.t. to euclidean distance to prototypes.
        """
        #self.prototypes.type(torch.DoubleTensor).to(self.support_features.device)
        query_features = self.compute_features(query_images).unsqueeze(0)   #.type(torch.DoubleTensor).to(self.support_features.device)
        clip_weights = clip_weights.unsqueeze(0)

        clip_weights = clip_weights.transpose(1,2)
        clip_logits = 100. * query_features @ clip_weights.transpose(1, 2)
        clip_logits = torch.softmax(clip_logits, dim=-1)

        num_classes = self.support_labels.unique().size(0)
        #support_labels_one_hot = nn.functional.one_hot(self.support_labels, num_classes) .type(torch.DoubleTensor).to(self.support_features.device).unsqueeze(0)
        support_labels_one_hot = get_one_hot(self.support_labels.unsqueeze(0)).type(torch.FloatTensor).to(self.support_features.device)
        #print(support_labels_one_hot.shape)
        self.N_s = support_labels_one_hot.shape[1]
        self.N_q = query_features.shape[1]

        self.init_weights(self.support_features.unsqueeze(0).type(torch.FloatTensor).cuda(), support_labels_one_hot, query_features, clip_logits)
        for i in range(self.fine_tuning_steps):
            P_q = self.get_logits(query_features).softmax(2)
            self.q_update(P=P_q, clip_logits=clip_logits)
            self.weights_update(self.support_features.unsqueeze(0), query_features, support_labels_one_hot)

        self.prototypes = self.weights[0]
        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features[0]),
            #self.get_logits(query_features),
            temperature=self.temperature,
        ).detach()

    @staticmethod
    def is_transductive() -> bool:
        return True