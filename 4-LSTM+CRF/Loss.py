import torch
import torch.nn as nn
import pdb


class CRFLoss(nn.Module):
    def __init__(self, device, start_tag_id, end_tag_id):
        super().__init__()
        self.to(device)
        self.device = device
        self.start_tag_id = start_tag_id
        self.end_tag_id = end_tag_id

    def forward(self, transition, target, reduction='mean', start_tag_id=None, end_tag_id=None):
        """
        :param transition: bsz, seq_len, num_tags, num_tags
        :param target: bsz, seq_len
        :param reduction: 'mean', 'none'
        :param start_tag_id:
        :param end_tag_id: end_tag_id
        :return:
        """
        if end_tag_id is None:
            end_tag_id = self.end_tag_id
        if start_tag_id is None:
            start_tag_id = self.start_tag_id

        bsz, seqlen, num_tags = transition.shape[0], transition.shape[1], transition.shape[-1]
        # pdb.set_trace()
        idxes = torch.tensor(range(bsz))
        target_score = transition[idxes, 0, start_tag_id, target[:, 0]]
        for t in range(1, seqlen):
            target_score += transition[idxes, t, target[:, t - 1], target[:, t]]
        # pdb.set_trace()
        # forw: bsz, seq_len, num_tags -> scrolling bsz, num_tags

        # forw = exp_transition[:, 0, start_tag_id, :].unsqueeze(1)
        forw = transition[:, 0, start_tag_id, :]  # bsz, num_tags
        eps = 1e-12
        for t in range(1, seqlen):
            tmp = forw.unsqueeze(-1).repeat(1, 1, num_tags) + transition[:, t, :, :]  # bsz, num_tags(src), num_tags(dst)
            max_bias = tmp.max(dim=1)[0]  # bsz, num_tags(dst)
            tmp -= max_bias.unsqueeze(1).repeat(1, num_tags, 1)  # bsz, num_tags(src), num_tags(dst)
            forw = (torch.exp(tmp).sum(dim=1) + eps).log() + max_bias

        #   forw = torch.bmm(forw, exp_transition[:, t, :, :])
        # bsz(1 x num_tags) @ bsz(num_tags x num_tags) -> bsz(1 x num_tags)
        # log_sum_exp_all_score = (forw[:, -1, end_tag_id]+eps).log() + a  # bsz, 1, num_tags
        # log_sum_exp_all_score = torch.where(torch.isnan(sum_exp_all_score) | torch.isinf(sum_exp_all_score),
        #                                torch.full_like(sum_exp_all_score, 1e300), sum_exp_all_score)

        loss = -target_score + forw[:, end_tag_id]
        if reduction == 'mean':
            # if torch.isnan(loss.mean()) or torch.isinf(loss.mean()):
            #     pdb.set_trace()
            return loss.mean()
        elif reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        else:
            assert 0, "Unknown reduction type"
