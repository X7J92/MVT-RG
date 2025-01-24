import torch


def cal_nll_loss2(logit, idx, mask, weights=None):
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        # [nb * nw, seq_len]
        nll_loss = (nll_loss * weights).sum(dim=-1)
    # nll_loss = nll_loss.mean()
    return nll_loss.contiguous()


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    logit = logit.log_softmax(dim=-1)


    index = idx.unsqueeze(-1).long().cuda()

    nll_loss = -logit.gather(dim=-1, index=index).squeeze(-1)  # [nb * nw, seq_len]

    smooth_loss = -logit.sum(dim=-1)  # [nb * nw, seq_len]
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    epsilon = 1e-9
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / (mask.sum(dim=-1)+epsilon)
    else:
        # [nb * nw, seq_len]
        nll_loss = (nll_loss * weights).sum(dim=-1)
    # nll_loss = nll_loss.mean()
    return nll_loss.contiguous()


def weakly_supervised_loss(props_align, words_logit, words_id, words_mask, rewards,sentence_mask,
                           weights=None, neg_words_logit=None):
    bz=sentence_mask.size(0)
    sentence_mask = sentence_mask.view(bz*8,1).expand(-1,3)

    # values = [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000]
    # tensor_on_gpu = torch.tensor(values, dtype=torch.float64).cuda()
    # rewards = tensor_on_gpu
    # num_proposals = rewards.size(0)
    words_logit = words_logit.log_softmax(dim=-1)
    nll_loss = cal_nll_loss(words_logit, words_id, words_mask)
    nll_loss = nll_loss.view(-1, 3)
    nll_loss[torch.isnan(nll_loss)] = 0



    idx = torch.argsort(nll_loss, dim=-1, descending=True)
    _, idx = torch.sort(idx, dim=-1)

    rewards = rewards[idx]

    prop_loss = -(rewards * props_align.cuda().log_softmax(dim=-1))

    nll_loss = nll_loss.mean()

    prop_loss = prop_loss*sentence_mask

    prop_loss = prop_loss.mean(dim=-1).mean(dim=-1)

    final_loss = nll_loss + 1.0 * prop_loss

    return  final_loss