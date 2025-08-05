import torch
import torch.nn as nn
import TIM_p2_solver.tim_p2_utils as lib

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def run_tim_p2(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):

    kwargs_adm = {
      "fine_tuning_steps": 150,
      "cross_entropy_weight": 0.1,
      "marginal_entropy_weight": 1.0,
      "conditional_entropy_weight": 0.1,
      "temperature": 15.0,
      "alpha": 1.0,
      "gamma": 1.0,
      "feature_normalization": 2
    }

    device = clip_weights.device
    fewshot_model = lib.TIM_p2(nn.Identity(), **kwargs_adm).to(device)

    print("\n-------- Searching hyperparameters on the val set. --------")
    neighbor_index = torch.argmax(val_features @ test_features.T, dim=1)
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
    val_labels = val_labels.cuda()

    temperature_list = [120]
    xentr_list = [0.4]
    centr_list = [0.1]
    gamma_list = [0.05]

    best_acc = -1
    best_t = 0
    best_xentr = 0
    best_centr = 0
    best_gamma = 0

    for  gamma in gamma_list:
        for temp in temperature_list:
            for xentr in xentr_list:
                for centr in centr_list:
                    kwargs_adm["temperature"] = temp
                    kwargs_adm["cross_entropy_weight"] = xentr
                    kwargs_adm["conditional_entropy_weight"] = centr
                    kwargs_adm["gamma"] = gamma
                    fewshot_model = lib.TIM_p2(nn.Identity(), **kwargs_adm).to(device)

                    fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
                    method_logits = fewshot_model(test_features, clip_weights).detach().data
                    acc = cls_acc(method_logits[neighbor_index, :], val_labels)
                    print("temperature = {} ; xentr = {}  ; centr = {} ; gamma = {}".format(temp, xentr, centr, gamma))
                    print("**** TIM++'s val accuracy: {:.2f}. ****\n".format(acc))
                    if acc > best_acc:
                        best_acc = acc
                        best_t = temp
                        best_xentr = xentr
                        best_centr = centr
                        best_gamma = gamma


    print("\n-------- Evaluating on the test set. --------")
    kwargs_adm["temperature"] = best_t
    kwargs_adm["cross_entropy_weight"] = best_xentr
    kwargs_adm["conditional_entropy_weight"] = best_centr
    kwargs_adm["gamma"] = best_gamma

    fewshot_model = lib.TIM_p2(nn.Identity(), **kwargs_adm).to(device)

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # TIM++
    fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
    method_logits = fewshot_model(test_features, clip_weights).detach().data

    acc = cls_acc(method_logits, test_labels)
    print("**** tim_p2_adm's test accuracy for seed {}: {:.2f}. ****\n".format(int(cfg["seed"]), acc))
    print("with temperature = {} ; xentr = {}  ; centr = {}  ; gamma = {} ; best val acc = {:.2f}".format(best_t, best_xentr, best_centr, best_gamma , best_acc))

    return None